import shutil
import sys
from typing import Dict, List

from server.query_process.base import query_time_extract
from server.query_process.query_analysis import query_ner

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
import uuid


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

    def do_search(self, query: str, top_k: int, score_threshold: float, embeddings: Embeddings, time_filter=False,
                  method="cos", **kwargs) -> List:
        # TODO score_threshold的逻辑还没想好怎么统一，暂时先不过滤
        if "bge-" in self.embed_model:
            if "zh" in self.embed_model:
                embeddings.query_instruction = "为这个句子生成表示以用于检索相关文章："
            else:
                embeddings.query_instruction = "Represent this sentence for searching relevant passages:"
        # if kwargs.get("method") == "cos":
        if method in ["cos", "knn", "hybrid", "keywords"]:
            return self.raw_doc_search(query=query, embedding=embeddings, method=method, top_k=top_k,
                                       time_filter=time_filter)
        else:
            return self.db[0].similarity_search_with_score(query, top_k)

    def do_add_doc(self, docs: List[Document], **kwargs, ) -> List[Dict]:
        from elasticsearch.helpers import bulk
        # ids = self.db[-1].add_documents(docs)
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        requests = []
        ids = [str(uuid.uuid4()) for _ in texts]
        embedding_model = self._load_embeddings()
        embeddings = embedding_model.embed_documents(list(texts))
        for i, text in enumerate(texts):
            metadata = metadatas[i] if metadatas else {}
            time = []
            if "publishTime" in metadata.keys():
                timeStamp = int(metadata["publishTime"])
                if len(str(timeStamp)) != 10:
                    timeStamp = timeStamp / pow(10, len(str(timeStamp)) - 10)
                import datetime
                dt_object = datetime.datetime.fromtimestamp(timeStamp)
                time.append(dt_object.timestamp())
            if "contentTime" in metadata.keys():
                for item in metadata["contentTime"]:
                    time.append(item)
            request = {
                "_op_type": "index",
                "_index": self.kb_name,
                "vector": embeddings[i],
                "text": text,
                "time": time,
                "metadata": metadata,
                "_id": ids[i],
            }
            requests.append(request)
        bulk(self.client, requests)
        self.client.indices.refresh(index=self.kb_name)
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
                       time_filter: bool = True) -> List:
        query_vector = embedding.embed_query(query)
        if method == "knn":
            query_body = generate_knn_query(text=query, embedding=embedding, size=top_k, time_filter=time_filter)
        elif method == "hybrid":
            query_body = generate_hybrid_query(text=query, embedding=embedding, size=top_k, time_filter=time_filter)
        elif method == "cos":
            query_body = generate_search_query(text=query, embedding=embedding, size=top_k, time_filter=time_filter)
        elif method == "keywords":
            query_body = generate_keywords_query(text=query, size=top_k, time_filter=time_filter)
        else:
            raise NotImplementedError
        response = self.client.search(index=self.kb_name, body=query_body)
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
        scroll_time = '1m'
        # 执行初始搜索请求
        response = self.client.search(index=self.kb_name, scroll=scroll_time, size=1000, body={})
        ans = []
        # 获取初始的滚动ID
        scroll_id = response['_scroll_id']
        # 遍历所有结果
        while True:
            # 处理当前批次的结果
            for hit in response["hits"]["hits"]:
                ans.append(
                    {key: value for key, value in hit["_source"].items() if key != "vector"})
            # 执行下一个滚动请求
            response = self.client.scroll(scroll_id=scroll_id, scroll=scroll_time)
            # 检查是否有更多结果
            if len(response['hits']['hits']) == 0:
                break
            # 更新滚动ID
            scroll_id = response['_scroll_id']
        return ans

    def allIndex(self):
        all_indices = self.client.indices.get_alias(index="*")
        print(all_indices)

    def getMapping(self):
        mapping = self.client.indices.get_mapping(index=self.kb_name)
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

    def get_doc(self):
        query = {
            "query": {
                "match_all": {}
            },
            "size": 10
        }
        response = self.client.search(index=self.kb_name, body=query)
        print(response)
        return response["hits"]["hits"]

    def find_doc(self, kb_file: KnowledgeFile, size=10):
        query = {
            "query": {
                "term": {
                    "metadata.source.keyword": kb_file.filepath
                }
            },
            "size": size
        }
        response = self.client.search(index=self.kb_name, body=query)
        hits = [hit for hit in response["hits"]["hits"]]
        result = []
        for hit in response["hits"]["hits"]:
            result.append(
                {key: value for key, value in hit["_source"].items() if key != "vector"})
        return result


if __name__ == "__main__":
    names = ["中国电力企业联合会-电网要闻", "习近平重要讲话数据库", "习近平重要讲话数据库test"]
    esService = ESKBService(names[2])
    # print(esService.getMapping())
    # kb_file = KnowledgeFile(filename="中国共产党第十九届中央委员会第六次全体会议公报.txt", knowledge_base_name=names[2])
    # print(esService.find_doc(kb_file=kb_file, size=100)[0])
    # file = Document(page_content="a nsdfsdf", metadata={"publishTime": 1546300800, "a": 2})
    # esService.do_add_doc([file])
    # print(esService.searchAll())
    # print(esService.search_docs(query="十九届六中全会精神", top_k=10, score_threshold=2))
    # print(esService.do_search(query="习近平最近行程", top_k=2, score_threshold=2,
    #                           embeddings=esService._load_embeddings(),
    #                           method="cos", time_filter=True))
    # esService.search_docs(query="习近平最近行程", time_filter=True)
    # text = "习近平2023年12月2日到2024年1月2日行程"
    esService.search_docs(query="习近平2021年11月8日行程", time_filter=True)
    # esService.get_doc()
    # import jionlp as jio
    # print(jio.ner.extract_time(text))