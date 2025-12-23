import clickhouse_connect
from openai import OpenAI
import pickle
import requests
import os
import time
import datetime
import re
import jieba
from collections import defaultdict
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    Docx2txtLoader,
    UnstructuredMarkdownLoader,
    UnstructuredExcelLoader,
    CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

class SimpleVLLMEmbeddings:
    # 保持不变，与您提供的代码一致
    def __init__(self, model, base_url, api_key='Empty'):
        self.model = model
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def embed_documents(self, texts):
        if not texts:
            return []
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
            )
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text):
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    def __call__(self, text):
        return self.embed_query(text)

class RAGSystem:
    def __init__(self, 
                 embedding_model, 
                 knowledge_base_dir, 
                 vllm_host,
                 clickhouse_config,  # 新增ClickHouse配置
                 dict_path="Empty",
                 force_rebuild=True,
                 chunk_size=4000,
                 chunk_overlap=100):
        """
        clickhouse_config 格式示例:
        {
            'host': 'localhost',
            'port': 8123,
            'username': 'default',
            'password': '',
            'database': 'default',
            'table': 'knowledge_base',
            'content_columns': ['content', 'description'],  # 需要索引的文本列
            'metadata_columns': ['source', 'title', 'id']    # 需要保留的元数据列
        }
        """
        self.knowledge_base_dir = knowledge_base_dir
        self.embeddings = SimpleVLLMEmbeddings(model=embedding_model, base_url=vllm_host)
        self.dict_path = dict_path
        self.force_rebuild = force_rebuild
        self.clickhouse_config = clickhouse_config
        
        # 向量存储
        self.vector_store = None
        self.documents = []
        self.bm25_index = None

        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            keep_separator=True
        )

        # 加载自定义词典
        if os.path.exists(self.dict_path):
            jieba.load_userdict(self.dict_path)
        
        # 初始化ClickHouse客户端
        self.ch_client = self._init_clickhouse_client()
        
        self.setup_knowledge_base(force_rebuild=self.force_rebuild)

    def _init_clickhouse_client(self):
        """初始化ClickHouse客户端"""
        try:
            return clickhouse_connect.get_client(
                host=self.clickhouse_config['host'],
                port=self.clickhouse_config['port'],
                username=self.clickhouse_config['username'],
                password=self.clickhouse_config['password'],
                database=self.clickhouse_config['database']
            )
        except Exception as e:
            print(f"ClickHouse连接失败: {e}")
            raise

    def setup_knowledge_base(self, force_rebuild=False):
        """初始化或加载知识库"""
        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        faiss_path = os.path.join(self.knowledge_base_dir, "faiss_index")
        bm25_path = os.path.join(self.knowledge_base_dir, "bm25_data.pkl")
        docs_path = os.path.join(self.knowledge_base_dir, "documents.pkl")

        # 检查是否需要重建索引
        if not force_rebuild and os.path.exists(faiss_path) and os.path.exists(bm25_path) and os.path.exists(docs_path):
            print("加载现有索引...")
            try:
                # 加载FAISS索引
                self.vector_store = FAISS.load_local(
                    faiss_path, self.embeddings, allow_dangerous_deserialization=True
                )
                
                # 加载BM25索引和文档
                with open(bm25_path, 'rb') as f:
                    bm25_data = pickle.load(f)
                    self.bm25_index = bm25_data['bm25_index']
                
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                    
                print(f"成功加载 {len(self.documents)} 个文档片段")
                return True
            except Exception as e:
                print(f"加载失败: {e}，尝试重建...")

        return self._build_indexes(faiss_path, bm25_path, docs_path)

    def _fetch_data_from_clickhouse(self):
        """从ClickHouse获取数据并转换为Document对象"""
        print("从ClickHouse获取数据...")
        
        # 构建查询语句
        content_columns = self.clickhouse_config['content_columns']
        metadata_columns = self.clickhouse_config.get('metadata_columns', [])
        all_columns = content_columns + metadata_columns
        
        query = f"""
        SELECT {', '.join(all_columns)}
        FROM {self.clickhouse_config['database']}.{self.clickhouse_config['table']}
        """
        
        try:
            # 执行查询
            result = self.ch_client.query(query)
            rows = result.result_rows
            columns = result.column_names
            
            print(f"成功获取 {len(rows)} 条记录")
            
            # 转换为Document对象
            documents = []
            for row in rows:
                metadata = {}
                content_parts = []
                
                for col_name, value in zip(columns, row):
                    # 处理内容列
                    if col_name in content_columns:
                        if value is not None:
                            content_parts.append(str(value))
                    # 处理元数据列
                    elif col_name in metadata_columns:
                        metadata[col_name] = str(value) if value is not None else ""
                
                # 合并所有内容列
                full_content = "\n".join(content_parts)
                if full_content.strip():  # 只添加非空内容
                    doc = Document(
                        page_content=full_content,
                        metadata=metadata
                    )
                    documents.append(doc)
            
            print(f"转换为 {len(documents)} 个文档对象")
            return documents
            
        except Exception as e:
            print(f"从ClickHouse获取数据失败: {e}")
            raise

    def _build_indexes(self, faiss_path, bm25_path, docs_path):
        """构建索引 - 从ClickHouse获取数据"""
        print("构建新索引...")
        
        # 1. 从ClickHouse获取原始文档
        all_docs = self._fetch_data_from_clickhouse()
        if not all_docs:
            print("警告：未获取到任何文档数据")
            return False

        # 2. 分割文档
        print("分割文档...")
        self.documents = self.text_splitter.split_documents(all_docs)
        print(f"分割后得到 {len(self.documents)} 个文档片段")

        # 3. 构建FAISS向量索引
        print("构建FAISS向量索引...")
        self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
        self.vector_store.save_local(faiss_path)
        print("FAISS索引构建完成")

        # 4. 构建BM25全文检索索引
        print("构建BM25全文检索索引...")
        tokenized_corpus = [jieba.lcut(doc.page_content) for doc in self.documents]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        # 保存BM25索引
        with open(bm25_path, 'wb') as f:
            pickle.dump({'bm25_index': self.bm25_index}, f)
        
        # 保存文档对象（用于BM25检索）
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        print("索引构建完成!")
        return True

    # 以下方法保持不变，与您提供的代码一致
    def vector_search(self, query, top_k=5, output_table_name=False):
        if not self.vector_store:
            print("错误：知识库尚未设置。请先运行 setup_knowledge_base()。")
            return []
        
        tokenized_query = jieba.lcut(query)
        combined = f"原始查询: {query}\n分词结果: {tokenized_query}"
        results = self.vector_store.similarity_search(combined, k=top_k)
        if output_table_name:
            table_names = search_table_name(results)
            return table_names
        return results

    def keyword_search(self, query, top_k=5, output_table_name=False):
        if not self.bm25_index:
            print("错误：知识库尚未设置。请先运行 setup_knowledge_base()。")
            return []
        
        tokenized_query = jieba.lcut(query)
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        bm25_results = [self.documents[i] for i in bm25_top_indices]
        if output_table_name:
            table_names = search_table_name(bm25_results)
            return table_names
        return bm25_results

    def hybrid_search(self, question, top_k=8, rrf_k=60, output_table_name=False):
        if not self.vector_store:
            print("错误：知识库尚未设置。请先运行 setup_knowledge_base()。")
            return []
        
        # 执行向量检索
        vector_results = self.vector_search(question, top_k=top_k*2)
        
        # 执行全文检索
        keyword_results = self.keyword_search(question, top_k=top_k*2)

        # RRF 融合
        print("正在进行RRF融合...")
        ranked_scores = defaultdict(float)
        
        # 处理向量检索结果
        for i, doc in enumerate(vector_results):
            doc_content = doc.page_content
            rank = i + 1
            ranked_scores[doc_content] += 1.0 / (rrf_k + rank)
            
        # 处理全文检索结果
        for i, doc in enumerate(keyword_results):
            doc_content = doc.page_content
            rank = i + 1
            ranked_scores[doc_content] += 1.0 / (rrf_k + rank)

        # 按RRF分数排序
        sorted_docs_content = sorted(ranked_scores.keys(), key=lambda x: ranked_scores[x], reverse=True)
        
        # 去重并保持顺序
        final_docs = []
        seen_content = set()
        all_retrieved_docs = vector_results + keyword_results
        
        for content in sorted_docs_content:
            if len(final_docs) >= top_k:
                break
            if content not in seen_content:
                seen_content.add(content)
                for doc in all_retrieved_docs:
                    if doc.page_content == content:
                        final_docs.append(doc)
                        break
        
        if output_table_name:
            table_names = search_table_name(final_docs)
            return table_names  
        return final_docs
    


def search_table_name(docs):
    # 检索相关文档
    # print(f"检索到的文档数: \n{len(docs)}"):
    table_names = [extract_table_name(doc) for doc in docs]
    return [name for name in table_names if name is not None]

def extract_table_name(doc):
    match = re.search(r'英文表名:\s*(\S+)', doc.page_content)
    return match.group(1) if match else None


def extract_doc_page_content(retrieval_results):
    """步骤1: Schema Linking (识别相关表)"""
    if not retrieval_results:
        context_str = "无相关信息"
    else:
        context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieval_results])
    return context_str