import os

class Config:
    # LLM 配置
    VLLM_HOST = "http://10.130.37.12:8000/v1"
    LLM_MODEL = "Qwen3-8B"

    EMBEDDING_HOST = "http://10.130.37.7:8000/v1"
    EMBEDDING_MODEL = "Qwen3-Embedding-8B"
    KNOWLEDGE_BASE_DIR = "my_knowledge_base/rag_kb/kb"
    KNOWLEDGE_BASE_EXAMPLE_LIBRARY_DIR = "my_knowledge_base/icl_knowledge_base"
    KNOWLEDGE_SQL_DIR = "my_knowledge_base/sql/入湖总表_.csv"

    # 数据库配置
    CK_HOST='127.0.0.1'  # 数据库主机地址
    CK_PORT=8123           # HTTP 接口端口（默认8123）
    CK_USERNAME='default'    # 用户名
    CK_PASSWORD='12345678'   # 密码（如果有）
    CK_DATABASE='sap'    # 默认数据库


if __name__ == "__main__":
    cfg = Config()
    print("Knowledge Base Directory:", cfg.KNOWLEDGE_BASE_DIR)
    print("ClickHouse Host:", cfg.CK_HOST)
    print("ClickHouse Port:", cfg.CK_PORT)