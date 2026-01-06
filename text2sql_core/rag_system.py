from typing import Optional
from openai import OpenAI
import pickle
import os
import re
import jieba
import uuid  # 新增：用于生成唯一ID
from collections import defaultdict
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, Docx2txtLoader, 
    UnstructuredMarkdownLoader, UnstructuredExcelLoader, CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from text2sql_core.table_info_extraction import load_table_meta_as_docs

# from pcode.sqlgen import extract_all_tables_info

class SimpleVLLMEmbeddings:
    def __init__(self, model, base_url, api_key='Empty'):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def embed_documents(self, texts):
        if not texts: return []
        # 注意：实际生产中建议分批次调用embedding接口，避免一次性请求过大
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(model=self.model, input=text)
            embeddings.append(response.data[0].embedding)
        return embeddings

    def embed_query(self, text):
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding

    def __call__(self, text):
        return self.embed_query(text)


class RAGSystem:
    def __init__(self, embedding_model, knowledge_base_dir, vllm_host, dict_path="Empty"):
        self.knowledge_base_dir = knowledge_base_dir
        self.embeddings = SimpleVLLMEmbeddings(model=embedding_model, base_url=vllm_host)
        self.dict_path = dict_path
        
        # 核心数据结构
        self.documents = []        # 统一的文档列表（包含metadata和id）
        self.doc_map = {}          # id -> Document 的映射，方便快速查找
        self.vector_store = None   # FAISS 索引
        self.bm25_index = None     # BM25 索引
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=100,
            keep_separator=True
        )

        if os.path.exists(self.dict_path):
            jieba.load_userdict(self.dict_path)
        
        self.loader_mapping = {
            ".txt": TextLoader,
            ".md": UnstructuredMarkdownLoader,
            ".pdf": PyPDFLoader,
            ".docx": Docx2txtLoader,
            ".xlsx": UnstructuredExcelLoader,
            ".xls": UnstructuredExcelLoader, 
            ".csv": lambda path: CSVLoader(path, encoding='gbk'), 
        }

    def _load_indexes_from_paths(self, faiss_path: str, data_path: str) -> bool:
        """
        从指定路径加载 FAISS 向量库和文档/BM25 数据。
        成功返回 True，失败返回 False。
        """
        if not (os.path.exists(faiss_path) and os.path.exists(data_path)):
            return False

        try:
            # 1. 加载向量库
            self.vector_store = FAISS.load_local(
                faiss_path, self.embeddings, allow_dangerous_deserialization=True
            )
            # 2. 加载文档和 BM25
            with open(data_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.bm25_index = data['bm25_index']
            # 3. 重建 ID 映射
            self.doc_map = {doc.metadata['chunk_id']: doc for doc in self.documents}
            return True
        except Exception as e:
            print(f"⚠️ 索引加载失败: {e}")
            return False

    def init_kb_from_files(self, file_paths, force_rebuild: bool=False) -> bool:
        """从文件初始化知识库。"""
        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        faiss_path = os.path.join(self.knowledge_base_dir, "faiss_index")
        data_path = os.path.join(self.knowledge_base_dir, "rag_data.pkl") # 统一存储文档和BM25
        if not force_rebuild and self._load_indexes_from_paths(faiss_path, data_path):
            print("✅ 知识库加载成功！")
            return True
        all_docs = []
        # 1. 加载文件
        for root, _, files in os.walk(file_paths):
            for file_name in files:
                ext = os.path.splitext(file_name)[-1].lower()
                if ext in self.loader_mapping:
                    try:
                        loader = self.loader_mapping[ext](os.path.join(root, file_name))
                        all_docs.extend(loader.load())
                    except Exception as e:
                        print(f"加载 {file_name} 失败: {e}")

        if not all_docs:
            print(f"没有找到可加载的文件。")
            return False
        return self._build_indexes(all_docs, faiss_path, data_path)

    def init_kb_from_clickhouse(self, ck_client, force_rebuild: bool = False) -> bool:
        """从 ClickHouse 数据库初始化知识库。"""
        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        faiss_path = os.path.join(self.knowledge_base_dir, "faiss_index")
        data_path = os.path.join(self.knowledge_base_dir, "rag_data.pkl")

        if not force_rebuild and self._load_indexes_from_paths(faiss_path, data_path):
            print("✅ 知识库加载成功！")
            return True
        table_info=load_table_meta_as_docs(ck_client)
        if not table_info:
            print("没有从 ClickHouse 中提取到表信息。")
            return False
        return self._build_indexes(table_info, faiss_path, data_path)

    def _build_indexes(self, documents, faiss_path, data_path):
        print("正在构建统一索引...")

        if not documents:
            return False

        print(f"原始文档数: {len(documents)}，正在分块...")
        self.documents = self.text_splitter.split_documents(documents)

        # 3. 分配 chunk_id
        for doc in self.documents:
            doc.metadata['chunk_id'] = str(uuid.uuid4())

        self.doc_map = {
            doc.metadata['chunk_id']: doc for doc in self.documents
        }

        print("构建 FAISS 向量库...")
        self.vector_store = FAISS.from_documents(
            self.documents, self.embeddings
        )
        self.vector_store.save_local(faiss_path)

        print("构建 BM25 索引...")
        tokenized_corpus = [
            jieba.lcut(doc.page_content) for doc in self.documents
        ]
        self.bm25_index = BM25Okapi(tokenized_corpus)

        with open(data_path, 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'bm25_index': self.bm25_index
            }, f)

        print("索引构建完成。")
        return True

    def vector_search(self, query, top_k=5, output_table_name=False):
        if not self.vector_store: return []
        
        tokenized_query = jieba.lcut(query)
        combined = f"原始查询: {query}\n分词结果: {tokenized_query}"
        
        results = self.vector_store.similarity_search(combined, k=top_k)
        
        if output_table_name:
            return search_table_name(results)
        return results

    def keyword_search(self, query, top_k=5, output_table_name=False):
        if not self.bm25_index: return []
        
        tokenized_query = jieba.lcut(query)
        # 获取 BM25 分数
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # 排序获取前 top_k 个索引
        top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        
        # 直接通过索引从 self.documents 获取对应的文档
        # 因为我们构建时保证了顺序一致，所以 index 0 的分数对应 self.documents[0]
        results = [self.documents[i] for i in top_indices]
        
        if output_table_name:
            return search_table_name(results)
        return results

    def hybrid_search(self, question, top_k=8, rrf_k=60, output_table_name=False):
        """
        基于 chunk_id 的精确混合检索 (RRF)
        """
        if not self.vector_store or not self.bm25_index:
            print("错误：索引未就绪")
            return []
        
        # 扩大检索范围以获得更好的交集
        search_k = top_k * 2
        
        # 1. 向量检索
        # 注意：此处返回的是 Document 对象列表
        vector_docs = self.vector_search(question, top_k=search_k)
        
        # 2. 关键词检索
        keyword_docs = self.keyword_search(question, top_k=search_k)

        # 3. RRF 融合计算
        # 使用 chunk_id 作为唯一键值，而不是 page_content
        rrf_score_map = defaultdict(float)
        
        # 处理向量结果
        for rank, doc in enumerate(vector_docs):
            doc_id = doc.metadata.get('chunk_id')
            if doc_id:
                rrf_score_map[doc_id] += 1.0 / (rrf_k + rank + 1)
            
        # 处理关键词结果
        for rank, doc in enumerate(keyword_docs):
            doc_id = doc.metadata.get('chunk_id')
            if doc_id:
                rrf_score_map[doc_id] += 1.0 / (rrf_k + rank + 1)

        # 4. 排序并取回文档
        # 按 RRF 分数倒序
        sorted_ids = sorted(rrf_score_map.keys(), key=lambda x: rrf_score_map[x], reverse=True)
        
        final_docs = []
        for doc_id in sorted_ids[:top_k]:
            # 从之前建立的 doc_map 中取回原始文档对象，确保 metadata 完整
            if doc_id in self.doc_map:
                final_docs.append(self.doc_map[doc_id])

        if output_table_name:
            return search_table_name(final_docs)
        return final_docs

# 辅助函数保持不变
def search_table_name(docs):
    table_names = [extract_table_name(doc) for doc in docs]
    return [name for name in table_names if name is not None]

def extract_table_name(doc):
    match = re.search(r'英文表名:\s*(\S+)', doc.page_content)
    return match.group(1) if match else None

def extract_doc_page_content(retrieval_results):
    if not retrieval_results:
        context_str = "无相关信息"
    else:
        context_str = "\n\n---\n\n".join([doc.page_content for doc in retrieval_results])
    return context_str