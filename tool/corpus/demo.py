import json
import os
import sys
sys.path.append("../../")
# from server.chat.chat import chat_local
# from server.knowledge_base.kb_service.es_kb_service import ESKBService
# from server.knowledge_base.utils import KnowledgeFile


def template():
    prompt = "请为我生成一些数据表明用户是否想要查询习近平重要讲话数据库或者普通对话，返回json数组格式，" + \
             "如[{\"query\" : 生成的用户查询, \"type\" : 0}, {\"query\" : 生成的用户查询, \"type\" : 1}]; 其中0表示普通对话， 1表示查询数据库\n" + \
             "json数组:"
    return prompt

def gen_IR_data():
    query = []
    with open("./corpus.jsonl", "r") as file:
        data = file.readlines()
        for item in data:
            temp = json.loads(item.strip())
            query.append(temp["query"])
    print(len(query))
    with open("./query_type_data.csv", "a+") as file:
        for item in query:
            file.write(item + ",&1\n")

def gen_IR_data2():
    query = []
    with open("/Users/rich/Downloads/PKUMOD-CCKS/训练集.txt", "r") as file:
        data = file.readlines()
        for item in data:
            if item[0] == "q":
                query.append(item[item.index(":") + 1:].split("\n")[0].strip())
    with open("/Users/rich/Downloads/SogouQA.json", 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data:
            query.append(item["question"])
    with open("/Users/rich/Downloads/train-zen-v1.0.json", 'r', encoding='utf-8') as file:
        data = json.load(file)
        for item in data["data"]:
            for paragraph in item["paragraphs"]:
                for qas in paragraph["qas"]:
                    query.append(qas["question"])
    print(len(query))
    with open("./query_type_data.csv", "a+") as file:
        for item in query:
            file.write(item + ",&0\n")




if __name__ == "__main__":
    gen_IR_data()
    gen_IR_data2()
