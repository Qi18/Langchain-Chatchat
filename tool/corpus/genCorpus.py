import json
from math import inf
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from server.knowledge_base.utils import KnowledgeFile
from server.chat.chat import chat_local
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.knowledge_base.kb_service.es_kb_service import ESKBService



def genCorpus(filepath):
    name = "习近平重要讲话数据库"
    esService = ESKBService(name)
    infos = esService.searchAll()
    hasSuc = set()
    if os.path.exists(filepath):
        with open(filepath, "r+") as file:
            for info in file.readlines():
                temp = json.loads(info.strip())
                hasSuc.add(temp["name"])
    ans = []
    for info in infos:
        print(info)
        if "_" in os.path.basename(info["metadata"]["source"]):
            continue
        filename = os.path.basename(info["metadata"]["source"]) + "__" + str(info["metadata"]["chunk_index"])
        if filename in hasSuc:
            continue

        result = chat_local(query=template(os.path.basename(info["metadata"]["source"]), info["text"], 5),
                            temperature=0.1)
        print(result)
        try:
            query_list = json.loads(result)
            ans.append({'name': filename, 'query': query_list["questions"]})
        except Exception as e:
            print(e)
            continue
        if len(ans) > 1:
            with open(filepath, "a+") as file:
                file.write('\n'.join([json.dumps(item, ensure_ascii=False) for item in ans]) + '\n')
                ans = []
    if len(ans) != 0:
        with open(filepath, "a+") as file:
            file.write('\n'.join([json.dumps(item, ensure_ascii=False) for item in ans]) + '\n')


def template(title, content, num: int = 5):
    print(content)
    prompt = \
        "请根据文档内容生成" + str(num) + \
        "个独立的问题。返回一个json如：{questions : ['question1', 'question2']}" + \
        "=======\n" + title + "\n" + content + "\n" + "========\n" + \
        "生成的json："
    return prompt


esService = ESKBService("习近平重要讲话数据库")
def data_change():
    origin_file = "/data/lrq/llm/sync/Langchain-Chatchat/tool/corpus/corpus1.jsonl"
    with open(origin_file, "r") as file:
        lines = file.readlines()
    ans = []
    file = open("/data/lrq/llm/sync/Langchain-Chatchat/tool/corpus/toy_finetune_data.jsonl", "w+")
    for line in lines:
        data = json.loads(line)
        for query in data["query"]:
            res = {}
            res["query"] = query
            res["pos"] = [query_es(data["name"])]
            res["neg"] = []
            ans.append(res)
            if len(ans) > 500:
                file.write('\n'.join([json.dumps(item, ensure_ascii=False) for item in ans]) + '\n')
                ans = []
    if len(ans) != 0:
        with open("toy_finetune_data.jsonl", "a+") as file:
            file.write('\n'.join([json.dumps(item, ensure_ascii=False) for item in ans]) + '\n')


def query_es(name):
    infos = esService.find_doc(kb_file=KnowledgeFile(filename=name.split("_")[0], knowledge_base_name="习近平重要讲话数据库"), size=max(10000, int(name.split("__")[1])))
    print(infos)
    for info in infos:
        print(info["metadata"]["chunk_index"])
        if info["metadata"]["chunk_index"] == int(name.split("__")[1]):
            print(f"find {name}")
            return info["content"]
    print(f"no {name}")



if __name__ == "__main__":
    filepath = "corpus.jsonl"
    genCorpus(filepath)
    # data_change()
    # query_es("实现中华民族伟大复兴.txt__0")

