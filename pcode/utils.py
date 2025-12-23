import asyncio
import json
import re
import logging
from pcode.sqlgen import query_ck
from config import Config

def parse_llm_json_output(llm_output: str) -> dict:
    """
    尝试从 LLM 的字符串输出中提取并解析 JSON 对象
    """
    try:
        # 1. 尝试直接解析
        return json.loads(llm_output)
    except json.JSONDecodeError:
        pass

    try:
        # 2. 使用正则提取第一个 { 和最后一个 } 之间的内容 (去除 markdown 标记)
        match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if match:
            json_str = match.group(0)
            return json.loads(json_str)
    except Exception as e:
        logging.error(f"JSON解析失败: {e} \n原始内容: {llm_output}")
    
    # 3. 解析失败时的兜底返回
    return {
        "status": "error",
        "sql": "",
        "result": None,
        "message": "模型返回格式无法解析，请重试。"
    }


# 异步包装 query_system.query_ck（类方法）
async def async_query_ck_method(query_system, user_question, all_tables_info, case_info, ck_client):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        query_system.query_ck,
        user_question,
        all_tables_info,
        case_info,
        3,  # max_retries
        ck_client
    )


# 异步包装全局 query_ck 函数
async def async_query_ck_func(ck_client, openai_client, llm_model, all_tables_info, user_question, enable_thinking=False):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        query_ck,                # ← 全局函数
        ck_client,
        openai_client,
        llm_model,
        all_tables_info,
        user_question,
        enable_thinking,
        3                        # max_retries
    )