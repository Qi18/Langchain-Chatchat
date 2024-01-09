import operator
from abc import ABC, abstractmethod

import os
from datetime import datetime

import numpy as np
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from sklearn.preprocessing import normalize

from server.db.repository.knowledge_base_repository import (
    add_kb_to_db, delete_kb_from_db, list_kbs_from_db, kb_exists,
    load_kb_from_db, get_kb_detail,
)
from server.db.repository.knowledge_file_repository import (
    add_file_to_db, delete_file_from_db, delete_files_from_db, file_exists_in_db,
    count_files_from_db, list_files_from_db, get_file_detail, delete_file_from_db,
    list_docs_from_db, add_files_to_db,
)

from configs import (kbs_config, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD,
                     EMBEDDING_MODEL, DEFAULT_VS_TYPE, RERANK_MODEL)
from server.knowledge_base.utils import (
    get_kb_path, get_doc_path, load_embeddings, KnowledgeFile,
    list_kbs_from_folder, list_files_from_folder
)
from server.query_process.base import query_time_extract
from server.utils import embedding_device, rerank_device
from typing import List, Union, Dict, Optional, Tuple, Any
from server.knowledge_base.kb_cache.rerank_model import load_rerank_model, ReRankModel
from loguru import logger


class SupportedVSType:
    FAISS = 'faiss'
    MILVUS = 'milvus'
    DEFAULT = 'default'
    PG = 'pg'
    ES = 'es'


class KBService(ABC):

    def __init__(self,
                 knowledge_base_name: str,
                 rerank_model: str = RERANK_MODEL,
                 embed_model: str = EMBEDDING_MODEL,
                 ):
        self.kb_name = knowledge_base_name
        self.embed_model = embed_model
        self.rerank_model = rerank_model
        self.kb_path = get_kb_path(self.kb_name)
        self.doc_path = get_doc_path(self.kb_name)
        self.do_init()

    def _load_embeddings(self, embed_device: str = embedding_device()) -> Embeddings:
        return load_embeddings(self.embed_model, embed_device)

    def _load_reranks(self, rerank_device: str = rerank_device()) -> ReRankModel:
        return load_rerank_model(self.rerank_model, rerank_device)

    def save_vector_store(self):
        '''
        保存向量库:FAISS保存到磁盘，milvus保存到数据库。PGVector暂未支持
        '''
        pass

    def create_kb(self):
        """
        创建知识库
        """
        if not os.path.exists(self.doc_path):
            os.makedirs(self.doc_path)
        self.do_create_kb()
        status = add_kb_to_db(self.kb_name, self.vs_type(), self.embed_model)
        return status

    def clear_vs(self):
        """
        删除向量库中所有内容
        """
        self.do_clear_vs()
        status = delete_files_from_db(self.kb_name)
        return status

    def drop_kb(self):
        """
        删除知识库
        """
        self.do_drop_kb()
        status = delete_kb_from_db(self.kb_name)
        return status

    def add_doc(self, kb_file: KnowledgeFile, **kwargs):
        """
        向知识库添加文件
        如果指定了docs，则不再将文本向量化，并将数据库对应条目标为custom_docs=True
        """
        docs = kb_file.file2text()
        self.delete_doc(kb_file)
        doc_infos = self.do_add_doc(docs, **kwargs)
        status = add_file_to_db(kb_file=kb_file, docs_count=len(docs), doc_infos=doc_infos)
        return status

    def add_docs(self, kb_files: List[KnowledgeFile], **kwargs):
        """
        一次上传多个文件，为了多文件上传数据加速，请尽量使用这个接口
        """
        all_docs = []
        db_datas = []

        for kb_file in kb_files:
            all_docs.extend(kb_file.file2text())
        if not all_docs:
            return True
        # 将所有的chunk写入向量库
        doc_infos = self.do_add_doc(all_docs, **kwargs)
        # chunk 分类
        db_file_info = {}
        for item in doc_infos:
            if item["metadata"]["source"] not in db_file_info:
                db_file_info[item["metadata"]["source"]] = []
            db_file_info[item["metadata"]["source"]].append(item)
        # 根据kb_file排序
        for kb_file in kb_files:
            # if db_file_info.get(kb_file.filepath) is None:
            #     continue
            db_datas.append(
                {"length_docs": len(db_file_info[kb_file.filepath]), "doc_infos": db_file_info[kb_file.filepath]})
        # 整体写入metadata数据库
        # kb_files和db_datas一一对应
        status = add_files_to_db(kb_files=kb_files, kb_data=db_datas)
        return status

    def delete_doc(self, kb_file: KnowledgeFile, delete_content: bool = False, **kwargs):
        """
        从知识库删除文件
        """
        self.do_delete_doc(kb_file, **kwargs)
        status = delete_file_from_db(kb_file)
        if delete_content and os.path.exists(kb_file.filepath):
            os.remove(kb_file.filepath)
        # logger.info(f"删除文档{self.kb_name}成功")
        return status

    def update_doc(self, kb_file: KnowledgeFile, **kwargs):
        """
        使用content中的文件更新向量库
        如果指定了docs，则使用自定义docs，并将数据库对应条目标为custom_docs=True
        """
        if os.path.exists(kb_file.filepath):
            self.delete_doc(kb_file, **kwargs)
        return self.add_doc(kb_file, **kwargs)

    def update_docs(self, kb_files: List[KnowledgeFile], **kwargs):
        for kb_file in kb_files:
            if os.path.exists(kb_file.filepath):
                self.delete_doc(kb_file, **kwargs)
        return self.add_docs(kb_files, **kwargs)

    def exist_doc(self, file_name: str):
        return file_exists_in_db(KnowledgeFile(knowledge_base_name=self.kb_name,
                                               filename=file_name))

    def list_files(self):
        return list_files_from_db(self.kb_name)

    def count_files(self):
        return count_files_from_db(self.kb_name)

    def search_docs(self,
                    query: str,
                    top_k: int = VECTOR_SEARCH_TOP_K,
                    score_threshold: float = SCORE_THRESHOLD,
                    search_method: str = "hybrid",
                    use_rerank: bool = True,
                    time_filter: bool = True,
                    ):
        embeddings = self._load_embeddings()
        rerank_model = self._load_reranks()
        docs = []
        top_k_1 = top_k * 5 if use_rerank else top_k
        # 召回阶段
        if search_method == "hybrid":
            docsCos = self.do_search(query=query, top_k=top_k_1, score_threshold=score_threshold,
                                     embeddings=embeddings,
                                     method="cos", time_filter=time_filter)
            docsBM25 = self.do_search(query=query, top_k=top_k_1, score_threshold=score_threshold,
                                      embeddings=embeddings,
                                      method="keywords", time_filter=time_filter)
            docs = []
            exist_doc = set()
            ## 去重
            for doc in docsCos:
                doc_id = doc[0].metadata["source"] + "_" + str(doc[0].metadata["chunk_index"])
                if doc_id not in exist_doc:
                    docs.append(doc)
                    exist_doc.add(doc_id)
            for doc in docsBM25:
                doc_id = doc[0].metadata["source"] + "_" + str(doc[0].metadata["chunk_index"])
                if doc_id not in exist_doc:
                    docs.append(doc)
                    exist_doc.add(doc_id)

        elif search_method == "cos":
            docs = self.do_search(query=query, top_k=top_k_1, score_threshold=score_threshold,
                                  embeddings=embeddings,
                                  method="cos", time_filter=time_filter)
        elif search_method == "keywords":
            docs = self.do_search(query=query, top_k=top_k_1, score_threshold=score_threshold,
                                  embeddings=embeddings,
                                  method="keywords", time_filter=time_filter)
        logger.info("召回阶段完成，一共召回了{}个结果".format(len(docs)))
        # logger.info("召回结果如下：")
        # for doc in docs:
        #     logger.info(doc[0])
        #     logger.info(doc[1])
        if not docs or not use_rerank:
            return docs[:top_k]

        # 排序阶段
        # if time_filter:
        #     out_time_query, time_words = query_time_extract(query)
        #     docs = rerank_model.rerank(docs, out_time_query, top_k)
        # else:
        docs = rerank_model.rerank(docs, query, top_k)
        return docs

    def search_docs_multiQ(self,
                           querys: list,  # 第一个是原始query，后面是multiquery
                           top_k: int = VECTOR_SEARCH_TOP_K,
                           score_threshold: float = SCORE_THRESHOLD,
                           search_method: str = "hybrid",
                           ):
        print(f"querys:{querys}")
        if len(querys) == 0:
            return []
        embeddings = self._load_embeddings()
        rerank_model = self._load_reranks()
        docs = []
        if search_method == "hybrid":
            docsCos = self.do_search(query=querys[0], top_k=top_k * 5, score_threshold=score_threshold,
                                     embeddings=embeddings,
                                     method="cos")
            docsBM25 = self.do_search(query=querys[0], top_k=top_k * 5, score_threshold=score_threshold,
                                      embeddings=embeddings,
                                      method="keywords")
            docs = docsCos
            ## 去重
            content = set([doc[0].page_content for doc in docs])
            for new_doc in docsBM25:
                if new_doc[0].page_content in content:
                    continue
                docs.append(new_doc)
                content.add(new_doc[0].page_content)
            for query in querys[1:]:
                for method in ["cos", "keywords"]:
                    new_docs = self.do_search(query=query, top_k=top_k * 5, score_threshold=score_threshold,
                                              embeddings=embeddings,
                                              method=method)
                    for new_doc in new_docs:
                        if new_doc[0].page_content in content:
                            continue
                        docs.append(new_doc)
                        content.add(new_doc[0].page_content)

        elif search_method == "cos" or search_method == "keywords":
            docs = self.do_search(query=querys[0], top_k=top_k * 5, score_threshold=score_threshold,
                                  embeddings=embeddings,
                                  method=search_method)
            content = set([doc[0].page_content for doc in docs])
            for query in querys[1:]:
                new_docs = self.do_search(query=query, top_k=top_k * 5, score_threshold=score_threshold,
                                          embeddings=embeddings,
                                          method=search_method)
                for new_doc in new_docs:
                    if new_doc[0].page_content in content:
                        continue
                    docs.append(new_doc)
                    content.add(new_doc[0].page_content)
        if not docs:
            return []
        return rerank_model.rerank(docs, querys[0], top_k)

    def get_doc_by_id(self, id: str) -> Optional[Document]:
        return None

    def list_docs(self, file_name: str = None, metadata: Dict = {}) -> List[Document]:
        '''
        通过file_name或metadata检索Document
        '''
        doc_infos = list_docs_from_db(kb_name=self.kb_name, file_name=file_name, metadata=metadata)
        docs = [self.get_doc_by_id(x["id"]) for x in doc_infos]
        return docs

    @abstractmethod
    def do_create_kb(self):
        """
        创建知识库子类实自己逻辑
        """
        pass

    @staticmethod
    def list_kbs_type():
        return list(kbs_config.keys())

    @classmethod
    def list_kbs(cls):
        return list_kbs_from_db()

    def exists(self, kb_name: str = None):
        kb_name = kb_name or self.kb_name
        return kb_exists(kb_name)

    @abstractmethod
    def vs_type(self) -> str:
        pass

    @abstractmethod
    def do_init(self):
        pass

    @abstractmethod
    def do_drop_kb(self):
        """
        删除知识库子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_search(self,
                  query: str,
                  top_k: int,
                  score_threshold: float,
                  embeddings: Embeddings,
                  **kwargs
                  ) -> List[Tuple[Document, Any]]:
        """
        搜索知识库子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_add_doc(self,
                   docs: List[Document],
                   ) -> List[Dict]:
        """
        向知识库添加文档子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_delete_doc(self,
                      kb_file: KnowledgeFile):
        """
        从知识库删除文档子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_clear_vs(self):
        """
        从知识库删除全部向量子类实自己逻辑
        """
        pass


class KBServiceFactory:

    @staticmethod
    def get_service(kb_name: str,
                    vector_store_type: Union[str, SupportedVSType],
                    embed_model: str = EMBEDDING_MODEL,
                    ) -> KBService:
        if isinstance(vector_store_type, str):
            vector_store_type = getattr(SupportedVSType, vector_store_type.upper())
        if SupportedVSType.FAISS == vector_store_type:
            from server.knowledge_base.kb_service.faiss_kb_service import FaissKBService
            return FaissKBService(kb_name, embed_model=embed_model)
        if SupportedVSType.PG == vector_store_type:
            from server.knowledge_base.kb_service.pg_kb_service import PGKBService
            return PGKBService(kb_name, embed_model=embed_model)
        elif SupportedVSType.MILVUS == vector_store_type:
            from server.knowledge_base.kb_service.milvus_kb_service import MilvusKBService
            return MilvusKBService(kb_name,
                                   embed_model=embed_model)  # other milvus parameters are set in model_config.kbs_config
        elif SupportedVSType.ES == vector_store_type:
            from server.knowledge_base.kb_service.es_kb_service import ESKBService
            return ESKBService(kb_name,
                               embed_model=embed_model)
        elif SupportedVSType.DEFAULT == vector_store_type:  # kb_exists of default kbservice is False, to make validation easier.
            from server.knowledge_base.kb_service.default_kb_service import DefaultKBService
            return DefaultKBService(kb_name)

    @staticmethod
    def get_service_by_name(kb_name: str
                            ) -> KBService:
        _, vs_type, embed_model = load_kb_from_db(kb_name)
        if vs_type is None and os.path.isdir(get_kb_path(kb_name)):  # faiss knowledge base not in db
            vs_type = DEFAULT_VS_TYPE
            embed_model = EMBEDDING_MODEL
        return KBServiceFactory.get_service(kb_name, vs_type, embed_model)

    @staticmethod
    def get_default():
        return KBServiceFactory.get_service("default", SupportedVSType.DEFAULT)


def get_kb_details() -> List[Dict]:
    kbs_in_folder = list_kbs_from_folder()
    kbs_in_db = KBService.list_kbs()
    result = {}

    for kb in kbs_in_folder:
        result[kb] = {
            "kb_name": kb,
            "vs_type": "",
            "embed_model": "",
            "file_count": 0,
            "create_time": None,
            "in_folder": True,
            "in_db": False,
        }

    for kb in kbs_in_db:
        kb_detail = get_kb_detail(kb)
        if kb_detail:
            kb_detail["in_db"] = True
            if kb in result:
                result[kb].update(kb_detail)
            else:
                kb_detail["in_folder"] = False
                result[kb] = kb_detail

    data = []
    for i, v in enumerate(result.values()):
        v['No'] = i + 1
        data.append(v)
    return data


def get_kb_file_details(kb_name: str) -> List[Dict]:
    kb = KBServiceFactory.get_service_by_name(kb_name)
    files_in_folder = list_files_from_folder(kb_name)
    files_in_db = kb.list_files()
    result = {}

    for doc in files_in_folder:
        result[doc] = {
            "kb_name": kb_name,
            "file_name": doc,
            "file_ext": os.path.splitext(doc)[-1],
            "file_version": 0,
            "document_loader": "",
            "docs_count": 0,
            "text_splitter": "",
            "create_time": None,
            "in_folder": True,
            "in_db": False,
        }
    for doc in files_in_db:
        doc_detail = get_file_detail(kb_name, doc)
        if doc_detail:
            doc_detail["in_db"] = True
            if doc in result:
                result[doc].update(doc_detail)
            else:
                doc_detail["in_folder"] = False
                result[doc] = doc_detail

    data = []
    for i, v in enumerate(result.values()):
        v['No'] = i + 1
        data.append(v)

    return data


class EmbeddingsFunAdapter(Embeddings):

    def __init__(self, embeddings: Embeddings):
        self.embeddings = embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return normalize(self.embeddings.embed_documents(texts))

    def embed_query(self, text: str) -> List[float]:
        query_embed = self.embeddings.embed_query(text)
        query_embed_2d = np.reshape(query_embed, (1, -1))  # 将一维数组转换为二维数组
        normalized_query_embed = normalize(query_embed_2d)
        return normalized_query_embed[0].tolist()  # 将结果转换为一维数组并返回

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await normalize(self.embeddings.aembed_documents(texts))

    async def aembed_query(self, text: str) -> List[float]:
        return await normalize(self.embeddings.aembed_query(text))


def score_threshold_process(score_threshold, k, docs):
    if score_threshold is not None:
        cmp = (
            operator.le
        )
        docs = [
            (doc, similarity)
            for doc, similarity in docs
            if cmp(similarity, score_threshold)
        ]
    return docs[:k]


def rerank_score_process(docs):
    return [doc for doc in docs if doc[1] > 0]
