import json
from server.knowledge_base.kb_service.es_kb_service import ESKBService
from server.knowledge_base.utils import KnowledgeFile

esService = ESKBService("习近平重要讲话数据库")


def data_change():
    origin_file = "corpus.jsonl"
    with open(origin_file, "r") as file:
        lines = file.readlines()
    ans = []
    file = open("toy_finetune_data.jsonl", "w+")
    for line in lines:
        res = {}
        data = json.loads(line)
        res["query"] = data["query"]
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
    for info in infos:
        if info["metadata"]["chunk_index"] == int(name.split("__")[1]):
            return info["content"]
    print(name)


if __name__ == "__main__":
    data_change()
