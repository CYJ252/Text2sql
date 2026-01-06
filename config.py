import os

class Config:
    # LLM 配置
    VLLM_HOST = os.getenv("VLLM_HOST", "http://10.130.37.12:8000/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "Qwen3-8B")

    EMBEDDING_HOST = os.getenv("EMBEDDING_HOST", "http://10.130.37.7:8000/v1")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-8B")
    KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "my_knowledge_base/rag_kb/kb")
    KNOWLEDGE_BASE_EXAMPLE_LIBRARY_DIR = os.getenv("KNOWLEDGE_BASE_EXAMPLE_LIBRARY_DIR", "my_knowledge_base/icl_knowledge_base")
    KNOWLEDGE_SQL_DIR = os.getenv("KNOWLEDGE_SQL_DIR", "my_knowledge_base/sql/入湖总表_.csv")

    # 数据库配置（从环境变量读取）
    CK_HOST = os.getenv("CK_HOST", "127.0.0.1")
    CK_PORT = int(os.getenv("CK_PORT", "8123"))
    CK_USERNAME = os.getenv("CK_USERNAME", "default")
    CK_PASSWORD = os.getenv("CK_PASSWORD", '12345678')  # 不设默认值！必须提供
    CK_DATABASE = os.getenv("CK_DATABASE", "sap")