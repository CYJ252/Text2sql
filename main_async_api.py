import asyncio
import time
from fastapi import FastAPI, HTTPException
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

# 全局客户端（在应用启动时初始化）
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

        rag_system = RAGSystem(
            embedding_model=EMBEDDING_MODEL,
            knowledge_base_dir=KNOWLEDGE_BASE_DIR,
            vllm_host=EMBEDDING_HOST,
            dict_path="my_knowledge_base/dict.txt",
        )
        # 初始化表名检索知识库
        rag_system.init_kb_from_clickhouse(ck_client, force_rebuild=False) 
        # rag_system.init_kb_from_files(file_paths=KNOWLEDGE_BASE_DIR,force_rebuild=True)

        ICL_system = RAGSystem(
            embedding_model=EMBEDDING_MODEL,
            knowledge_base_dir=KNOWLEDGE_BASE_EXAMPLE_LIBRARY_DIR, #检索文件夹存放地址，文件最好用csv格式
            vllm_host=EMBEDDING_HOST,
            dict_path="my_knowledge_base/dict.txt",
        )
        # 初始化案例检索知识库
        ICL_system.init_kb_from_files(file_paths=KNOWLEDGE_BASE_EXAMPLE_LIBRARY_DIR,force_rebuild=True)



        # 初始化查询系统
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
@app.post("/query", response_model=QueryResponse)
async def text_to_sql_query(request: QueryRequest):
    try:
        user_question = request.question.strip()
        if not user_question:
            raise HTTPException(status_code=400, detail="问题不能为空")
        start_time = time.time()
        # 1. 表名检索
        table_names = rag_system.hybrid_search(user_question, top_k=15, output_table_name=True)
        all_tables_info = extract_some_tables_info(ck_client, table_names, sample_num=5)

        # 2. 案例检索
        case_docs = icl_system.vector_search(user_question, top_k=5)
        case_info = extract_doc_page_content(case_docs)

        #✅ 3. 并发执行两个 SQL 生成与查询链路
        task1 = async_query_ck_method(query_system, user_question, all_tables_info, case_info, ck_client)
        task2 = async_query_ck_func(
            ck_client=ck_client,
            openai_client=openai_client,
            llm_model=LLM_MODEL,
            all_tables_info=all_tables_info,
            user_question=user_question,
            enable_thinking=False
        )
        # 等待两个任务同时完成
        (sql1, result1), (sql2, result2) = await asyncio.gather(task1, task2)

        # sql1, result1 = query_system.query_ck(user_question, all_tables_info, case_info, ck_client=ck_client)
        # sql2, result2 = query_ck(
        #     ck_client=ck_client,
        #     openai_client=openai_client,
        #     llm_model=LLM_MODEL,
        #     all_tables_info=all_tables_info,
        #     user_question=user_question,
        #     enable_thinking=False
        # )  


        print('----------- SQL1 ------------')
        print(sql1)
        print('---------- RESULT1 ----------')
        print(result1)
        print('----------- SQL2 -----------')
        print(sql2)
        print('---------- RESULT2 ----------')
        print(result2)

        sql_time = time.time()
        print(f"sql生成时间: {sql_time - start_time:.2f} 秒")

        # 4. 后续处理（如融合、总结等）
        final_json_dict = query_system.generate_json_analysis(user_question, sql1, result1, sql2, result2)
        end_time = time.time()
        print(f"总查询时间: {end_time - start_time:.2f} 秒")
        return QueryResponse(**final_json_dict)

    except Exception as e:
        logging.error(f"查询出错: {e}", exc_info=True)
        return QueryResponse(
            status="error",
            sql="",
            result=None,
            message=f"处理请求时发生内部错误: {str(e)}"
        )
    
