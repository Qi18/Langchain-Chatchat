import json
import os.path
import sys
sys.path.append("../")
import numpy as np
from matplotlib.font_manager import FontProperties
from tqdm import tqdm

from configs import logger, log_verbose
from server.knowledge_base.kb_service.es_kb_service import ESKBService
import pandas as pd
import matplotlib.pyplot as plt
from pyecharts.charts import Bar3D


def cal_MRR(corpus_datas: list,
            esService: ESKBService,
            search_method: str = "cos",
            use_rerank: bool = False,):
    MRR = [0.0] * 3
    Recall = [0.0] * 3
    if search_method == "hybrid" and not use_rerank:
        return MRR, Recall
    index = 0
    for data in tqdm(corpus_datas, desc="语料query", leave=True):
        docs = esService.search_docs(query=data["query"], top_k=10, search_method=search_method,
                                     use_rerank=use_rerank)
        for j in range(2):
            for i in range(pow(10, j)):
                try:
                    filename = os.path.splitext(os.path.basename(docs[i][0].metadata["source"]))[0]
                except Exception as e:
                    msg = f"加载文档 {filename} 时出错：{e}"
                    logger.error(f'{e.__class__.__name__}: {msg}',
                                 exc_info=e if log_verbose else None)
                if filename == data["name"] or filename == (data["name"] + "_summary"):
                    MRR[j] += 1.0 / (i + 1)
                    Recall[j] += 1
                    break
        index += 1
        print(search_method + " " + str(use_rerank) + " MRR 1 " + str(MRR[0] / index) + "\n")
        print(search_method + " " + str(use_rerank) + " MRR 10 " + str(MRR[1]/ index)+ "\n")
        print(search_method + " " + str(use_rerank) + " Recall 1 " + str(Recall[0]/ index) + "\n")
        print(search_method + " " + str(use_rerank) + " Recall 10 " + str(Recall[1]/ index) + "\n")

    MRR = [item / len(corpus_datas) for item in MRR]
    Recall = [item / len(corpus_datas) for item in Recall]
    return MRR, Recall


def draw_graph(corpus_datas: list,
               esService: ESKBService,
               use_rerank: bool = False):
    # 创建数据
    search_method = ["cos", "keywords", "hybrid"]
    top_k = ["top1", 'top10', "top100"]
    MRR = []
    Recall = []
    # with open('./cache.txt', 'r') as f:
    #     for line in f.readlines():
    #         MRR.append(json.loads(line.strip()))
    for i in tqdm(search_method, desc="search方法"):
        MRR_value, Recall_value = cal_MRR(corpus_datas, esService, search_method=i, use_rerank=use_rerank)
        MRR.append(MRR_value)
        Recall.append(Recall_value)

    plt.figure(figsize=(10, 6))
    tab = plt.table(cellText=MRR,
                    colLabels=top_k,
                    rowLabels=search_method,
                    loc='center',
                    cellLoc='center',
                    rowLoc='center')
    tab.scale(1, 2)
    plt.axis('off')
    plt.title("MRR of rerank" if use_rerank else "MRR of no rerank")
    plt.savefig("MRR_of_rerank.png" if use_rerank else "MRR_of_no_rerank.png")
    plt.show()

    plt.figure(figsize=(10, 6))
    tab = plt.table(cellText=Recall,
                    colLabels=top_k,
                    rowLabels=search_method,
                    loc='center',
                    cellLoc='center',
                    rowLoc='center')
    tab.scale(1, 2)
    plt.axis('off')
    plt.title("Recall of rerank" if use_rerank else "Recall of no rerank")
    plt.savefig("Recall_of_rerank.png" if use_rerank else "Recall_of_no_rerank.png")
    plt.show()


if __name__ == "__main__":
    names = ["中国电力企业联合会-电网要闻", "习近平重要讲话数据库"]
    esService = ESKBService(names[1])
    corpus_datas = []
    file = open("./corpus.txt", "r")
    for item in file.readlines():
        data = json.loads(item)
        corpus_datas.append(data)
    file.close()

    # draw_graph(corpus_datas, esService, use_rerank=False)
    draw_graph(corpus_datas, esService, use_rerank=True)
