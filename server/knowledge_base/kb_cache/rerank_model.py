from datetime import datetime, timedelta

import numpy as np
import torch
from FlagEmbedding import FlagReranker
from configs import RERANK_MODEL, MODEL_PATH
from server.utils import rerank_device
from loguru import logger
import os
import jionlp as jio
import time


class ReRankModel:
    def __init__(self, model=RERANK_MODEL, device=rerank_device()):
        self.model = model
        self.path = MODEL_PATH["rerank_model"][model]
        self.device = device
        self.rerank_model = None
        self.time_weight = 0.4
        self.model_weight = 1

    def _load_reranks(self, model: str = RERANK_MODEL, device: str = rerank_device()):
        if self.rerank_model is None:
            self.rerank_model = FlagReranker(self.path,
                                             use_fp16=False)  # Setting use_fp16 to True speeds up computation with a slight performance degradation

    # 时间query,领域知识的考虑
    def rerank(self, docs, query, top_k):
        document_map = {doc[0].page_content: doc[0] for doc in docs}
        rerank_pairs = [(query, doc[0].page_content) for doc in docs]
        reranked_list = self.rerank_by_model(rerank_pairs)
        parse_info = jio.ner.extract_time(query, time_base=time.time(), with_parsing=True)
        if parse_info:
            logger.info("query带有时间信息")
            time_docs = []
            out_time_docs = []
            for item in reranked_list:
                model_score = sigmoid(item[1])
                doc = document_map[item[0]]
                if "publishTime" in doc.metadata.keys() and doc_in_queryTime(parse_info,
                                                                             doc.metadata["publishTime"]):
                    time_docs.append((doc, model_score))
                else:
                    out_time_docs.append((doc, model_score))
            logger.info(f"满足时间的doc个数{len(time_docs)}")
            out_time_docs = self.domain_sort(out_time_docs)
            if len(time_docs) < top_k:
                time_docs.extend(out_time_docs[top_k - len(time_docs):])
            return time_docs[:top_k]
        else:
            logger.info("query不带有时间信息")
            doc_score = []
            for item in reranked_list:
                model_score = sigmoid(item[1])
                doc = document_map[item[0]]
                doc_score.append((doc, model_score))
            doc_score = self.domain_sort(doc_score)
            return doc_score[:top_k]
        # doc_score1 = sorted(doc_score, key=lambda x: -x[2])
        # logger.debug("model排序")
        # for item in doc_score1[:3]:
        #     logger.info(item[0])
        #     logger.info("总:" + str(item[1]))
        #     logger.info("model:" + str(item[2]))
        #     logger.info("addition:" + str(item[3]))
        # doc_score = sorted(doc_score, key=lambda x: -x[1])
        # logger.debug("总排序")
        # for item in doc_score[:3]:
        #     logger.info(item[0])
        #     logger.info("总:" + str(item[1]))
        #     logger.info("model:" + str(item[2]))
        #     logger.info("addition:" + str(item[3]))

    def rerankOnlyModel(self, docs, query, top_k):
        document_map = {doc[0].page_content: doc[0] for doc in docs}
        rerank_pairs = [(query, doc[0].page_content) for doc in docs]
        reranked_list = self.rerank_by_model(rerank_pairs)
        doc_score = []
        for item in reranked_list:
            model_score = item[1]
            doc = document_map[item[0]]
            doc_score.append((doc, sigmoid(model_score)))
        doc_score = sorted(doc_score, key=lambda x: -x[1])
        return doc_score[:top_k]

    # 使用rerank模型对候选答案进行排序
    def rerank_by_model(self, pairs):
        self._load_reranks()
        scores = self.rerank_model.compute_score(pairs)
        sorted_list = sorted(zip([ans[1] for ans in pairs], scores), key=lambda x: -x[1])
        return sorted_list

    def domain_sort(self, doc_score):
        # 百科排在最前面
        baike_domain = []
        other_domain = []
        for item in doc_score:
            if "domain" in item[0].metadata.keys() and item[0].metadata["domain"] in ["baike.baidu.com"]:
                baike_domain.append(item)
            else:
                other_domain.append(item)
        # 剩下的按照时间排序
        logger.info(f"百科个数{len(baike_domain)}")
        other_domain = self.time_sort(other_domain)
        baike_domain.extend(other_domain)
        return baike_domain

    def time_sort(self, doc_score):
        # 时间越近越好，加上model的分数
        ans = []
        for item in doc_score:
            if "publishTime" in item[0].metadata.keys():
                ans.append((item[0], item[1] * self.model_weight + cal_near_time_score(
                    item[0].metadata["publishTime"]) * self.time_weight))
            else:
                ans.append((item[0], item[1] * self.model_weight))
        logger.info(f"按最近时间+model打分排序{len(ans)}")
        return sorted(ans, key=lambda x: -x[1])


# 保证model预测的值域和addition_score的值域一致
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 判断doc的时间是否在query的时间范围内
def doc_in_queryTime(parse_info, timeInfo):
    timeInfo = timeInfo / 1000
    for item in parse_info:
        if item["type"] == "time_span" or item["type"] == "time_point":
            startTime = time.mktime(time.strptime(item["detail"]["time"][0], "%Y-%m-%d %H:%M:%S"))
            endTime = time.mktime(time.strptime(item["detail"]["time"][1], "%Y-%m-%d %H:%M:%S"))
            if startTime <= timeInfo <= endTime:
                return True
    return False


def cal_score(doc, query):
    time_weight = 0.8
    if "domain" in doc.metadata.keys() and "publishTime" in doc.metadata.keys():
        time_score = cal_time_score(doc.metadata["publishTime"], query)
        domain_score = cal_domain_score(doc.metadata["domain"])
        # logger.info(f"时间得分{time_score}")
        # logger.info(f"domain得分{domain_score}")
        return time_score * time_weight + (1 - time_weight) * domain_score
    elif "domain" not in doc.metadata.keys():
        return cal_time_score(doc.metadata["publishTime"], query) * time_weight
    else:
        return cal_domain_score(doc.metadata["domain"]) * (1 - time_weight)


def cal_near_time_score(timeStamp):
    assert len(str(timeStamp)) == 13, "时间戳是13位的"
    timeStamp = timeStamp / 1000
    time_interval = pow(10, 8)
    return max(1 - (int(time.time()) - timeStamp) / time_interval, -1)


# score都是小于1的
def cal_time_score(timeStamp, query):
    # query = query_time_process(query)
    # 给的timeStamp应该是毫秒级的 ，时间戳是13位的
    # TODO 测试时间戳间隔的长度
    # logger.info("时间戳的长度是{}".format(len(str(timeStamp))))
    assert len(str(timeStamp)) == 13, "时间戳是13位的"
    timeStamp = timeStamp / 1000
    time_interval = pow(10, 8)
    try:
        timeinfo = jio.parse_time(query, time_base=time.time())
    except Exception as e:
        # 用户query没有时间，按照最近的时间排序
        return max(1 - (int(time.time()) - timeStamp) / time_interval, -1)
    if timeinfo["type"] == "time_span":
        startTime = time.mktime(time.strptime(timeinfo["time"][0], "%Y-%m-%d %H:%M:%S"))
        endTime = time.mktime(time.strptime(timeinfo["time"][1], "%Y-%m-%d %H:%M:%S"))
        if startTime <= timeStamp:
            if endTime >= timeStamp:
                return 1
            else:
                return -1
        else:
            return -1
    elif timeinfo["type"] == "time_point":
        for time_point in timeinfo["time"]:
            # 将时间戳转换为datetime对象
            dt1 = datetime.strptime(time_point, "%Y-%m-%d %H:%M:%S")
            dt2 = datetime.fromtimestamp(timeStamp)
            # 计算两个datetime对象之间的时间间隔
            delta = dt2 - dt1
            # 检查时间间隔是否不超过一天
            if delta <= timedelta(days=1):
                return 1
        return -1
    else:
        return max(1 - (int(time.time()) - timeStamp) / time_interval, -1)


def cal_domain_score(domain):
    score_map = {"jhsjk.people.cn": 0.8, "baike.baidu.com": 1}
    return score_map[domain]


modelPool = {}


def load_rerank_model(model: str, device: str):
    if str not in modelPool.keys():
        modelPool[str] = ReRankModel(model, device)
    return modelPool[str]


if __name__ == "__main__":
    pass
