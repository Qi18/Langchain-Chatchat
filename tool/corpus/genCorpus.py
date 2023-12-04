import json
import os
import sys

sys.path.append("../../")
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
        if "_" in os.path.basename(info.metadata["source"]):
            continue
        filename = os.path.basename(info.metadata["source"]) + "__" + str(info.metadata["chunk_index"])
        if filename in hasSuc:
            continue

        result = chat_local(query=template(os.path.basename(info.metadata["source"]), info.page_content, 5),
                            temperature=0.9)
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


if __name__ == "__main__":
    filepath = "corpus1.jsonl"
    genCorpus(filepath)
