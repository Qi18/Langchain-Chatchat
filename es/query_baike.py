from elasticsearch import Elasticsearch, helpers, exceptions
import os
import jieba.posseg as pseg
from configs.server_config import DEFAULT_BIND_HOST
from server.knowledge_base.kb_service.es_utils import generate_knn_query, generate_hybrid_query, generate_search_query, \
    _default_knn_setting, generate_keywords_query, es_params, host, es_client


def delIndex(es, indexName):
    if es.indices.exists(index=indexName):
        res = es.indices.delete(index=indexName)
        print(res)


def searchAll(es):
    query = {
        "query": {
            "match_all": {}
        }
    }
    response = es.search(index="baike", body=query)
    print(response)


def countAll(es, indexName):
    count = es.count(index=indexName)['count']
    print(count)


def searchRelatedContent(query: str,
           indexName: str = "baike"):
    es = es_client
    # 分词query
    # analysis = es.indices.analyze(index=indexName, body={"text": query, "analyzer": "ik_max_word"}, )
    # tokens = [token['token'] for token in analysis['tokens']]
    words = pseg.cut(query)
    tokens = []
    for word, flag in words:
        print(word + ":" + flag)
        saveWords = ['nr', 'ns', 'nt', 'nz']
        for element in saveWords:
            if element in flag:
                tokens.append(word)
                break
    realTokens = tokens
    print("分析query的出来需要查询的词汇是：")
    print(realTokens)
    ans = []
    if len(realTokens) == 0:
        return ans
    # 构造query
    boolList = []
    # 联合查询
    for token in realTokens:
        boolList.append(
            {
                "match": {
                    "name.keyword": {
                        "query": token,
                        "boost": 5
                    }
                }
            })
        boolList.append(
            {
                "match": {
                    "name": {
                        "query": token,
                        "boost": 3
                    }
                }
            })
        boolList.append(
            {
                "match": {
                    "content": token
                }
            })
    query_body = {
        "query": {
            "bool": {
                "should": boolList
            }
        },
        "size": len(realTokens) * 3,
    }
    response = es.search(index=indexName, body=query_body)
    for res in response['hits']['hits']:
        if len(res['_source']['content']) < 5:
            continue
        ans.append({"name": res['_source']['name'], "score": res['_score'], "content": res['_source']['content']})
    # if len(realTokens) > 1:
    #     for token in realTokens:
    #         query_body = {
    #             "query": {
    #                 "bool": {
    #                     "should": [{
    #                         "match": {
    #                             "name.keyword": {
    #                                 "query": token,
    #                                 "boost": 5
    #                             }
    #                         }}, {
    #                         "match": {
    #                             "name": {
    #                                 "query": token,
    #                                 "boost": 3
    #                             }
    #                         }},{
    #                         "match": {
    #                             "content": token
    #                         }
    #                     }]
    #                 }
    #             },
    #             "size": 4,  # 返回前5个最相似的文档，可以根据需要调整
    #             # "_source": ["your_field_name"]  # 返回的字段，可以根据需要调整
    #         }
    #         response = es.search(index=indexName, body=query_body)
    #         for res in response['hits']['hits']:
    #             if len(res['_source']['content']) < 5:
    #                 continue
    #             ans.append(
    #                 {"name": res['_source']['name'], "score": res['_score'], "content": res['_source']['content']})
    return ans

if __name__ == "__main__":
    query = "习近平什么时候当选为党的总书记"
    for i in searchRelatedContent(query):
        print(i)
