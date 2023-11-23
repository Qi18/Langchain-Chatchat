import json

from elasticsearch import helpers
import jieba.posseg as pseg
from tqdm import tqdm

from server.knowledge_base.es_service.es_utils import es_client


def addBaike(indexName="baike"):
    es = es_client
    if es.indices.exists(index=indexName):
        print("index already exists")
        return
    mapping = {
        "mappings": {
            "properties": {
                "name": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_max_word",
                    "fields": {
                        "keyword": {
                            "type": "keyword"
                        }
                    }
                },
                "content": {
                    "type": "text",
                    "analyzer": "ik_max_word",
                    "search_analyzer": "ik_max_word"
                }
            }
        }
    }
    res = es.indices.create(index=indexName, body=mapping)
    print(res)
    upload_data = []
    progress_bar = tqdm(total=3000000, unit='iteration')
    index = 0
    with open("./563w_baidubaike.json", "r", encoding="utf-8") as file:
        while True:
            try:
                if file.readline() == "":
                    break
                data = json.loads(file.readline())
            except Exception as e:
                break
            index += 1
            upload_data.append({"_id": index, "name": data["title"], "content": data["summary"]})
            for item in data["sections"]:
                index += 1
                upload_data.append({"_id": index, "name": data["title"] + "_" + item["title"], "content": item["content"]})
            progress_bar.update(1)
            if len(upload_data) > 5000:
                helpers.bulk(es, upload_data, index=indexName)
                upload_data = []
        if len(data) != 0:
            helpers.bulk(es, upload_data, index=indexName)

def delIndex(indexName):
    if es_client.indices.exists(index=indexName):
        res = es_client.indices.delete(index=indexName)
        print(res)


def searchAll(indexName="baike"):
    query = {
        "query": {
            "match_all": {}
        }
    }
    response = es_client.search(index=indexName, body=query)
    print(response)


def countAll(es, indexName):
    count = es.count(index=indexName)['count']
    print(count)


def searchRelatedContent(query: str,
                         indexName: str = "baike"):
    es = es_client
    # 分词query
    # analysis = es_service.indices.analyze(index=indexName, body={"text": query, "analyzer": "ik_max_word"}, )
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
    ans = []
    if len(realTokens) == 0:
        realTokens.append(query)
    print(realTokens)
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
        "size": len(realTokens) * 10,
    }
    response = es.search(index=indexName, body=query_body)
    for res in response['hits']['hits']:
        if len(res['_source']['content']) < 5:
            continue
        ans.append({"name": res['_source']['name'], "score": res['_score'], "content": res['_source']['content']})
    return ans


if __name__ == "__main__":
    query = "十九届六中全会精神"
    for i in searchRelatedContent(query):
        print(i)