from elasticsearch import Elasticsearch, helpers, exceptions
import os
import jieba.posseg as pseg


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
    response = es.search(index="wiki", body=query)
    print(response)


def countAll(es, indexName):
    count = es.count(index=indexName)['count']
    print(count)


def search(indexName: "wiki", query):
    es = Elasticsearch([{'host': '127.0.0.1', 'port': 9200}])
    # 分词query
    # analysis = es.indices.analyze(index=indexName, body={"text": query, "analyzer": "ik_max_word"}, )
    # tokens = [token['token'] for token in analysis['tokens']]
    words = pseg.cut(query)
    tokens = []
    for word, flag in words:
        if 'n' in flag:
            tokens.append(word)
    # 去除停用词
    # stopwordsFile = "./all.stopwords"
    # stopwords = set()
    # with open(stopwordsFile, 'r', encoding='utf-8') as file:
    #     content = file.readlines()
    #     for line in content:
    #         stopwords.add(line.replace("\n", ""))
    # realTokens = []
    # for token in tokens:
    #     if token not in stopwords and len(token) > 1:
    #         realTokens.append(token)
    realTokens = tokens
    print(realTokens)
    ans = []
    # 构造query
    boolList = []
    for token in realTokens:
        boolList.append(
            {
                "match": {
                    "title": {
                        "query": token,
                        "boost": 10
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
        "size": 10,  # 返回前5个最相似的文档，可以根据需要调整
        # "_source": ["your_field_name"]  # 返回的字段，可以根据需要调整
    }
    response = es.search(index=indexName, body=query_body)
    for res in response['hits']['hits']:
        ans.append({"name": res['_source']['name'], "score": res['_score'], "content": res['_source']['content']})
    if len(realTokens) > 1:
        for token in realTokens:
            query_body = {
                "query": {
                    "bool": {
                        "should": [{
                            "match": {
                                "title": {
                                    "query": token,
                                    "boost": 10
                                }
                            }}, {
                            "match": {
                                "content": token
                            }
                        }]
                    }
                },
                "size": 2,  # 返回前5个最相似的文档，可以根据需要调整
                # "_source": ["your_field_name"]  # 返回的字段，可以根据需要调整
            }
            response = es.search(index=indexName, body=query_body)
            for res in response['hits']['hits']:
                ans.append(
                    {"name": res['_source']['name'], "score": res['_score'], "content": res['_source']['content']})
    return ans
