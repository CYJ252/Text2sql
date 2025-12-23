import asyncio
import json
import time
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging

# ====== 导入你已有的模块 ======
from config import Config
from pcode.sqlgen import extract_some_tables_info, query_ck
from text2sql_core.rag_system import RAGSystem, extract_doc_page_content
from text2sql_core.query_system import QuerySystem
import clickhouse_connect
from openai import OpenAI

from pcode.utils import async_query_ck_func, async_query_ck_method

# ====== 配置参数======
VLLM_HOST = Config.VLLM_HOST
LLM_MODEL = Config.LLM_MODEL

EMBEDDING_HOST = Config.EMBEDDING_HOST
EMBEDDING_MODEL = Config.EMBEDDING_MODEL

KNOWLEDGE_BASE_DIR = Config.KNOWLEDGE_BASE_DIR
KNOWLEDGE_BASE_EXAMPLE_LIBRARY_DIR = Config.KNOWLEDGE_BASE_EXAMPLE_LIBRARY_DIR

# 全局客户端
openai_client = None
ck_client = None
rag_system = None
icl_system = None
query_system = None

# ====== FastAPI App ======
app = FastAPI(
    title="Text2SQL API",
    description="基于 RAG 的自然语言转 SQL 查询系统",
    version="1.0.0"
)

# ====== 请求/响应模型 ======
from pcode.schemas import QueryRequest, QueryResponse


# ====== 启动事件：初始化全局资源 ======
@app.on_event("startup")
async def startup_event():
    global openai_client, ck_client, rag_system, icl_system, query_system
    try:
        # 初始化 OpenAI 客户端
        openai_client = OpenAI(base_url=VLLM_HOST, api_key="EMPTY")
        
        # 初始化 ClickHouse 客户端
        ck_client = clickhouse_connect.get_client(
            host=Config.CK_HOST,
            port=Config.CK_PORT,
            username=Config.CK_USERNAME,
            password=Config.CK_PASSWORD,
            database=Config.CK_DATABASE
        )

        # 初始化 RAG 系统
        rag_system = RAGSystem(
            embedding_model=EMBEDDING_MODEL,
            knowledge_base_dir=KNOWLEDGE_BASE_DIR,
            vllm_host=EMBEDDING_HOST,
            dict_path="my_knowledge_base/dict.txt",
            force_rebuild=False
        )

        icl_system = RAGSystem(
            embedding_model=EMBEDDING_MODEL,
            knowledge_base_dir=KNOWLEDGE_BASE_EXAMPLE_LIBRARY_DIR,
            vllm_host=EMBEDDING_HOST,
            dict_path="my_knowledge_base/dict.txt",
            force_rebuild=False
        )

        query_system = QuerySystem(
            llm_model=LLM_MODEL,
            vllm_host=VLLM_HOST,
            api_key="API_KEY",
        )

        logging.info("✅ 所有组件初始化成功！")

    except Exception as e:
        logging.error(f"❌ 初始化失败: {e}")
        raise RuntimeError(f"Startup failed: {e}")

# ====== 核心查询接口 ======
@app.get("/query") 
async def text_to_sql_query(question: str = Query(..., description="用户提出的问题")):
    user_question = question.strip()
    if not user_question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    async def event_generator():
        start_time = time.time()
        
        try:
            # --- 1. 发送处理中状态 ---
            yield create_sse_event("status", {
                "msg": "正在分析问题并查询数据...",
                "step": "sql_generation"
            })

            # ====== 阶段 1: 同步执行 SQL ======
            # 1. 检索表
            table_names = rag_system.hybrid_search(user_question, top_k=15, output_table_name=True)
            all_tables_info = extract_some_tables_info(ck_client, table_names, sample_num=5)

            # 2. 检索案例
            case_docs = icl_system.vector_search(user_question, top_k=5)
            case_info = extract_doc_page_content(case_docs)

            # 3. 生成并执行 SQL
            sql_query, result_markdown = query_system.query_ck(user_question, all_tables_info, case_info, ck_client=ck_client)
            
            # --- 2. 发送 SQL 执行结果 ---
            if result_markdown:
                yield create_sse_event("sql_result", {
                    "sql": sql_query,
                    "status": "success",
                    "table_data": result_markdown 
                })
            else:
                yield create_sse_event("sql_result", {
                    "sql": sql_query,
                    "status": "empty",
                    "table_data": None
                })
                # 无数据则发送结束信号并退出
                yield create_sse_event("done", {"reason": "no_data"})
                return

            # ====== 阶段 2: 流式生成分析报告 ======
            
            # 缓冲区，用于检测分隔符和积累 JSON
            buffer = ""
            json_sent = False
            separator = "#####REPORT_START#####"
            
            # 通知前端：开始分析
            yield create_sse_event("status", {"msg": "正在生成分析...", "step": "llm_analysis"})

            for chunk in query_system.generate_json_analysis_stream(user_question, sql_query, result_markdown):
                
                # 如果还没有发送过 JSON (即还在第一阶段)
                if not json_sent:
                    buffer += chunk
                    
                    # 检测是否出现了分隔符
                    if separator in buffer:
                        # 1. 切分 JSON 和 报告正文
                        parts = buffer.split(separator)
                        json_str = parts[0].strip()
                        report_start = parts[1] # 分隔符后面可能已经紧跟了一些报告文字
                        
                        # 2. 尝试解析并发送图表数据
                        try:
                            # 简单的清洗，防止 LLM 加了 ```json
                            json_str = json_str.replace("```json", "").replace("```", "").strip()
                            chart_data = json.loads(json_str)
                            
                            # 发送特定的 chart 事件
                            yield create_sse_event("chart_config", chart_data)
                        except Exception as e:
                            print(f"JSON解析失败: {e}")
                            # 失败也没关系，继续流程
                        
                        json_sent = True
                        
                        # 3. 如果分隔符后有残留的报告文字，作为 delta 发送
                        if report_start:
                            yield create_sse_event("delta", {"content": report_start})
                else:
                    # JSON 已发送，剩下的全是 Markdown 报告内容，直接流式透传
                    yield create_sse_event("delta", {"content": chunk})

            yield create_sse_event("delta", {"content": "#FINNSH#"})
        except Exception as e:
            # 发送错误事件
            yield create_sse_event("error", {
                "message": str(e),
                "detail": "后端处理异常"
            })
            logging.error(f"Streaming Error: {e}", exc_info=True)

    # 去掉了 headers 参数，仅保留 media_type
    return StreamingResponse(event_generator(), media_type="text/event-stream")
    

# ====== SSE 格式化辅助函数 ======
def create_sse_event(event_type: str, data: dict) -> str:
    """
    构造符合 SSE 标准的消息帧。
    格式:
    id: <unique_id>
    event: {event_type}
    data: {json_string}
    \n\n
    """
    # 将 data 字典转为 JSON 字符串，ensure_ascii=False 支持中文
    id ='0'  # 这里可以根据需要生成唯一 ID
    json_payload = json.dumps(data, ensure_ascii=False)
    
    # 构造标准 SSE 帧
    # 注意：event 字段是可选的，但加上会让前端处理更清晰
    # data 字段必须单行，json.dumps 会自动处理内部换行符的转义
    return f"event_id:{id}\nevent_type: {event_type}\ndata: {json_payload}\n\n"