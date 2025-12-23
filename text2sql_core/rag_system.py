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
    def __init__(self, embedding_model, knowledge_base_dir, vllm_host ,dict_path="Emoty",force_rebuild=True):
        self.knowledge_base_dir = knowledge_base_dir
        self.embeddings = SimpleVLLMEmbeddings(model=embedding_model, base_url=vllm_host)
        self.dict_path = dict_path
        self.force_rebuild = force_rebuild
        
        # 向量存储
        self.vector_store = None
        self.documents = []
        self.bm25_index = None

        # 全文检索的倒排索引
        self.inverted_index = defaultdict(list)
        
        # 文本分割器（建议使用RecursiveCharacterTextSplitter，更稳健）
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=100,
            keep_separator=True
        )

        # 加载自定义词典（如果有）
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
        self.setup_knowledge_base(force_rebuild=self.force_rebuild)

    def setup_knowledge_base(self, force_rebuild=False):
        """初始化或加载知识库"""
        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        faiss_path = os.path.join(self.knowledge_base_dir, "faiss_index")
        bm25_path = os.path.join(self.knowledge_base_dir, "bm25_data.pkl")

        if not force_rebuild and os.path.exists(faiss_path) and os.path.exists(bm25_path):
            print("加载现有索引...")
            try:
                self.vector_store = FAISS.load_local(
                    faiss_path, self.embeddings, allow_dangerous_deserialization=True
                )
                with open(bm25_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data['documents']
                    self.bm25_index = data['bm25_index']
                return True
            except Exception as e:
                print(f"加载失败: {e}，尝试重建...")

        return self._build_indexes(faiss_path, bm25_path)

    def _build_indexes(self, faiss_path, bm25_path):
        print("构建新索引...")
        all_docs = []
        for root, _, files in os.walk(self.knowledge_base_dir):
            for file_name in files:
                ext = os.path.splitext(file_name)[-1].lower()
                if ext in self.loader_mapping:
                    try:
                        loader = self.loader_mapping[ext](os.path.join(root, file_name))
                        all_docs.extend(loader.load())
                    except Exception as e:
                        print(f"加载 {file_name} 失败: {e}")

        if not all_docs:
            return False

        self.documents = self.text_splitter.split_documents(all_docs)

        # FAISS
        self.vector_store = FAISS.from_documents(self.documents, self.embeddings)
        self.vector_store.save_local(faiss_path)

        # BM25
        tokenized_corpus = [jieba.lcut(doc.page_content) for doc in self.documents]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        with open(bm25_path, 'wb') as f:
            pickle.dump({'documents': self.documents, 'bm25_index': self.bm25_index}, f)

        return True

    def vector_search(self, query, top_k=5,output_table_name=False):
        """
        执行向量检索
        """
        if not self.vector_store:
            print("错误：知识库尚未设置。请先运行 setup_knowledge_base()。")
            return []
        
        # print("执行向量检索...")
        tokenized_query = jieba.lcut(query)
        combined = f"原始查询: {query}\n分词结果: {tokenized_query}"
        results = self.vector_store.similarity_search(combined, k=top_k)
        if output_table_name:
            table_names = search_table_name(results)
            return table_names
        return results

    def keyword_search(self, query, top_k=5,output_table_name=False):
        """
        执行全文检索
        """
        if not self.bm25_index:
            print("错误：知识库尚未设置。请先运行 setup_knowledge_base()。")
            return []
        
        # print("执行全文检索...")
        # 2. BM25检索
        tokenized_query = jieba.lcut(query)
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        bm25_top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        bm25_results = [self.documents[i] for i in bm25_top_indices]
        if output_table_name:
            table_names = search_table_name(bm25_results)
            return table_names
        return bm25_results


    def hybrid_search(self, question, top_k=8, rrf_k=60,output_table_name=False):
        """
        执行混合检索并使用RRF进行结果融合
        """
        if not self.vector_store:
            print("错误：知识库尚未设置。请先运行 setup_knowledge_base()。")
            return []
        
        # 执行向量检索...
        vector_results = self.vector_search(question, top_k=top_k*2)
        
        # 执行全文检索...
        keyword_results = self.keyword_search(question, top_k=top_k*2)

        # RRF 融合
        print("正在进行RRF融合...")
        ranked_scores = defaultdict(float)
        
        # 处理向量检索结果
        for i, doc in enumerate(vector_results):
            # 使用 page_content 作为唯一标识符
            doc_content = doc.page_content
            rank = i + 1
            ranked_scores[doc_content] += 1.0 / (rrf_k + rank)
            
        # 处理全文检索结果
        for i, doc in enumerate(keyword_results):
            doc_content = doc.page_content
            rank = i + 1
            # 如果文档已存在，则累加分数；否则，创建新条目
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
                # 找到原始的Document对象（为了保留metadata）
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