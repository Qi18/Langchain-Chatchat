import shutil
import sys
from typing import Dict, List

sys.path.append("../../../")
from elasticsearch import Elasticsearch
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from loguru import logger

from server.knowledge_base.kb_service.base import KBService, SupportedVSType, KBServiceFactory
from server.knowledge_base.es_service.es_utils import generate_knn_query, generate_hybrid_query, generate_search_query, \
    _default_knn_setting, generate_keywords_query, es_params, host, es_client
from server.knowledge_base.utils import KnowledgeFile
from rich import print


class ESKBService(KBService):
    es_params: Dict
    host: str
    client: Elasticsearch
    db: List[ElasticsearchStore]

    def vs_type(self) -> str:
        return SupportedVSType.ES

    def do_init(self):
        self.es_params = es_params
        self.host = host
        self.client = es_client  # Elasticsearch([self.host], http_auth=(self.es_params['user'], self.es_params['password']))
        self.db = []
        embeddings = self._load_embeddings()
        if "bge-" in self.embed_model:
            if "zh" in self.embed_model:
                embeddings.query_instruction = "为这个句子生成表示以用于检索相关文章："
            else:
                embeddings.query_instruction = "Represent this sentence for searching relevant passages:"
            self.db.append(
                ElasticsearchStore(index_name=self.kb_name, embedding=embeddings, es_connection=self.client))
            embeddings.query_instruction = ""
        self.db.append(ElasticsearchStore(index_name=self.kb_name, embedding=embeddings, es_connection=self.client))
        logger.info(self.client.info())

    def do_search(self, query: str, top_k: int, score_threshold: float, embeddings: Embeddings, **kwargs) -> List:
        # TODO score_threshold的逻辑还没想好怎么统一，暂时先不过滤
        if "bge-" in self.embed_model:
            if "zh" in self.embed_model:
                embeddings.query_instruction = "为这个句子生成表示以用于检索相关文章："
            else:
                embeddings.query_instruction = "Represent this sentence for searching relevant passages:"
        if kwargs.get("method") == "cos":
            return self.raw_doc_search(query=query, embedding=embeddings, method="cos", top_k=top_k)
        elif kwargs.get("method") == "knn":
            return self.raw_doc_search(query=query, embedding=embeddings, method="knn", top_k=top_k)
        elif kwargs.get("method") == "hybrid":
            return self.raw_doc_search(query=query, embedding=embeddings, method="hybrid", top_k=top_k)
        elif kwargs.get("method") == "keywords":
            return self.raw_doc_search(query=query, embedding=embeddings, method="keywords", top_k=top_k)
        else:
            return self.db[0].similarity_search_with_score(query, top_k)

    def do_add_doc(self, docs: List[Document], **kwargs, ) -> List[Dict]:
        ids = self.db[-1].add_documents(docs)
        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        logger.info(f"添加文档至向量库{self.kb_name}成功")
        return doc_infos

    def do_delete_doc(self, kb_file: KnowledgeFile
                      , **kwargs):
        response = self.client.delete_by_query(
            index=self.kb_name,
            body={
                "query": {
                    "term": {
                        'metadata.source.keyword': kb_file.filepath
                    }
                }
            }
        )
        # print(response)

    def do_clear_vs(self):
        if self.client.indices.exists(index=self.kb_name):
            self.client.indices.delete(index=self.kb_name)
            logger.info(f"删除知识库{self.kb_name}成功")

    def disconnect(self):
        self.client.close()

    def do_create_kb(self):
        if not self.client.indices.exists(index=self.kb_name):
            embeddings = self._load_embeddings()
            dims = len(embeddings.embed_query("test"))
            setting = _default_knn_setting(dims)
            self.client.indices.create(index=self.kb_name, body=setting)
            logger.info(f"创建知识库{self.kb_name}成功")
        else:
            logger.error(f"知识库{self.kb_name}已经存在")

    def do_drop_kb(self):
        self.clear_vs()
        shutil.rmtree(self.kb_path)

    def raw_doc_search(self,
                       query: str,
                       embedding: Embeddings,
                       method: str = "cos",
                       top_k: int = 100,
                       knn_boost: float = 0.5):
        result = []
        query_vector = embedding.embed_query(query)
        if method == "knn":
            query_body = generate_knn_query(vec=query_vector, size=top_k)
        elif method == "hybrid":
            query_body = generate_hybrid_query(text=query, vec=query_vector, size=top_k)
        elif method == "cos":
            query_body = generate_search_query(vec=query_vector, size=top_k)
        elif method == "keywords":
            query_body = generate_keywords_query(text=query, size=top_k)
        else:
            query_body = generate_keywords_query(text=query, size=top_k)
        response = self.client.search(index=self.kb_name, body=query_body)
        hits = [hit for hit in response["hits"]["hits"]]
        for i in hits:
            result.append({
                'content': i['_source']['text'],
                'source': i['_source']['metadata']['source'],
                'score': i['_score']
            })
        # print(result)
        docs_and_scores = []
        for hit in response["hits"]["hits"]:
            docs_and_scores.append(
                (
                    Document(
                        page_content=hit["_source"]['text'],
                        metadata=hit["_source"]["metadata"],
                    ),
                    hit["_score"],
                )
            )
        return docs_and_scores

    def searchAll(self):
        query = {
            "query": {
                "match_all": {}
            }
        }
        response = self.client.search(index=self.kb_name, body=query)
        # print(response)
        return response

    def allIndex(self):
        all_indices = self.client.indices.get_alias(index="*")
        print(all_indices)

    def getMapping(self,
                   index_name):
        mapping = self.client.indices.get_mapping(index=index_name)
        print(mapping)

    def exist_doc(self, file_name: str):
        query = {
            "query": {
                "term": {
                    "metadata.source.keyword": file_name
                }
            }
        }
        response = self.client.search(index=self.kb_name, body=query)
        print(response)
        return response["hits"]["total"]["value"]

    def find_doc(self, kb_file: KnowledgeFile):
        query = {
            "query": {
                "term": {
                    "metadata.source.keyword": kb_file.filepath
                }
            }
        }
        response = self.client.search(index=self.kb_name, body=query)
        hits = [hit for hit in response["hits"]["hits"]]
        result = []
        for i in hits:
            result.append({
                'content': i['_source']['text'],
            })
        return result


if __name__ == "__main__":
    names = ["中国电力企业联合会-电网要闻", "习近平重要讲话数据库"]
    # esService = KBServiceFactory.get_service_by_name(names[1])
    esService = ESKBService(names[1])
    # kb_file = KnowledgeFile(filename="习近平复信美中航空遗产基金会主席和飞虎队老兵.txt", knowledge_base_name=names[1]
    #                         , chunk_overlap=100, chunk_size=400, metadata={"time": 1, "a": 2})
    # kb_file.file2text()
    print(esService.searchAll())
    # print(esService.search_docs(query="党的七大什么时间在哪里召开", top_k=10, score_threshold=2))
    # print(esService.searchAll())
