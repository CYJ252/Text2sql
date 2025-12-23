from config import Config
from pcode.sqlgen import extract_some_tables_info, query_ck
from text2sql_core.rag_system import RAGSystem, extract_doc_page_content
from text2sql_core.query_system import QuerySystem
import os
import re
from datetime import datetime
from openai import OpenAI
import clickhouse_connect
import jieba
import json


# ====== 配置参数======
VLLM_HOST = Config.VLLM_HOST
LLM_MODEL = Config.LLM_MODEL

EMBEDDING_HOST = Config.EMBEDDING_HOST
EMBEDDING_MODEL = Config.EMBEDDING_MODEL

KNOWLEDGE_BASE_DIR = Config.KNOWLEDGE_BASE_DIR
KNOWLEDGE_BASE_EXAMPLE_LIBRARY_DIR = Config.KNOWLEDGE_BASE_EXAMPLE_LIBRARY_DIR

openai_client = OpenAI(
    base_url=VLLM_HOST,
    api_key="EMPTY",
)

llm_model = LLM_MODEL
# 连接 ClickHouse 服务器
ck_client = clickhouse_connect.get_client(
    host=Config.CK_HOST,
    username=Config.CK_USERNAME,
    port=Config.CK_PORT,
    password=Config.CK_PASSWORD,
    database=Config.CK_DATABASE
)

def main():
    # 初始化RAG系统
    rag_system = RAGSystem(
        embedding_model=EMBEDDING_MODEL,
        knowledge_base_dir=KNOWLEDGE_BASE_DIR,
        vllm_host=EMBEDDING_HOST,
        dict_path="my_knowledge_base/dict.txt",
        force_rebuild=False
    )# 如果文件有修改需要强制重建知识库将force_rebuild设为True，或使用下面的函数
    # rag_system.setup_knowledge_base(force_rebuild=True) 


    ICL_system = RAGSystem(
        embedding_model=EMBEDDING_MODEL,
        knowledge_base_dir=KNOWLEDGE_BASE_EXAMPLE_LIBRARY_DIR, #检索文件夹存放地址，文件最好用csv格式
        vllm_host=EMBEDDING_HOST,
        dict_path="my_knowledge_base/dict.txt",
        force_rebuild=True
    )# 如果文件有修改需要强制重建知识库将force_rebuild设为True，或使用下面的函数
    # ICL_system.setup_knowledge_base(force_rebuild=True)


    # 初始化查询系统
    query_system = QuerySystem(
        llm_model=LLM_MODEL,
        vllm_host=VLLM_HOST,
        api_key="API_KEY",
    )

    while True:
        user_question = input("\n请输入你的问题 (输入 'exit' 或 'quit' 退出): ")
        if user_question.lower() in ['exit', 'quit']:
            break
        if not user_question.strip():
            continue
        # 表名检索，可以使用向量检索、关键词检索和混合检索
        results = rag_system.hybrid_search(user_question, top_k=10,output_table_name=True)
        all_tables_info = extract_some_tables_info(ck_client, results, sample_num=5)
        # 案例检索，可以使用向量检索、关键词检索和混合检索
        case_query = ICL_system.vector_search(user_question, top_k=5)
        case_info = extract_doc_page_content(case_query)

        sql, reuslt=query_system.query_ck(user_question, all_tables_info,case_info,ck_client=ck_client)

        # sql_query, rows = query_ck(
        #     ck_client=ck_client,
        #     openai_client=openai_client,
        #     llm_model=llm_model,
        #     all_tables_info=all_tables_info,
        #     user_question=user_question,
        #     enable_thinking=False
        # )

        print('----------- SQL1------------')
        print(sql)
        print('---------- RESULT1 ----------')
        print(reuslt)
        # print('----------- SQL2 -----------')
        # print(sql_query)
        # print('---------- RESULT2 ----------')
        # print(rows)

        # 生成最终的 JSON 分析结果（使用双sql）
        # final_json_dict = query_system.generate_json_analysis_2(user_question, sql, reuslt,sql_query,rows)

        # 生成最终的 JSON 分析结果（使用单sql）
        final_json_dict = query_system.generate_json_analysis(user_question, sql, reuslt)

        print('---------- JSON 结果 ----------')
        print(json.dumps(final_json_dict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
