from datetime import datetime, timedelta

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
        self.addition_weight = 0.2
        self.model_weight = 1

    def _load_reranks(self, model: str = RERANK_MODEL, device: str = rerank_device()):
        if self.rerank_model is None:
            self.rerank_model = FlagReranker(self.path,
                                             use_fp16=False)  # Setting use_fp16 to True speeds up computation with a slight performance degradation

    # 时间权重、模型权重、领域权重共同排序
    def rerank(self, docs, query, top_k):
        document_map = {doc[0].page_content: doc[0] for doc in docs}
        # score_map = {doc[0].page_content: doc[1] for doc in docs} #召回分数
        score_map = {}
        rerank_pairs = [(query, doc[0].page_content) for doc in docs]
        reranked_list = self.rerank_by_model(rerank_pairs)
        doc_score = []
        for item in reranked_list:
            model_score = item[1]
            doc = document_map[item[0]]
            addition_score = cal_score(doc, query)
            logger.info(f"模型得分{model_score},额外信息得分{addition_score}")
            doc_score.append((doc,
                              model_score * self.model_weight + addition_score * self.addition_weight))
        doc_score = sorted(doc_score, key=lambda x: -x[1])[:top_k]
        logger.info(f"rerank模型排序完成，共{len(doc_score)}个结果")
        logger.info(f"tok{top_k}结果为")
        logger.info(doc_score)
        return doc_score

    # 使用rerank模型对候选答案进行排序
    def rerank_by_model(self, pairs):
        self._load_reranks()
        scores = self.rerank_model.compute_score(pairs)
        sorted_list = sorted(zip([ans[1] for ans in pairs], scores), key=lambda x: -x[1])
        return sorted_list


def cal_score(doc, query):
    time_weight = 0.5
    if "domain" in doc.metadata.keys() and doc.metadata["domain"] == "baike.baidu.com":
        return 1
    if "domain" in doc.metadata.keys() and "publicTime" in doc.metadata.keys():
        return cal_time_score(doc.metadata["publicTime"], query) * time_weight + (1 - time_weight) * cal_domain_score(
            doc.metadata["domain"])
    elif "domain" not in doc.metadata.keys():
        return cal_time_score(doc.metadata["publicTime"], query) * time_weight
    else:
        return cal_domain_score(doc.metadata["domain"]) + (1 - time_weight)


# score都是小于1的
def cal_time_score(timeStamp, query):
    # 给的timeStamp应该是毫秒级的 ，时间戳是13位的
    # TODO 测试时间戳间隔的长度
    logger.info("时间戳的长度是{}".format(len(str(timeStamp))))
    timeStamp = timeStamp / 1000
    try:
        timeinfo = jio.parse_time(query, time_base=time.time())
    except Exception as e:
        # 用户query没有时间，按照最近的时间排序
        return 1 - (int(time.time()) - timeStamp) / pow(10, 8)
    if timeinfo["type"] == "time_span":
        startTime = time.mktime(time.strptime(timeinfo["time"][0], "%Y-%m-%d %H:%M:%S"))
        endTime = time.mktime(time.strptime(timeinfo["time"][1], "%Y-%m-%d %H:%M:%S"))
        if startTime <= timeStamp:
            if endTime >= timeStamp:
                return 1
            else:
                return 1 - (timeStamp - endTime) / pow(10, 8)
        else:
            return 1 - (startTime - timeStamp) / pow(10, 8)
    elif timeinfo["type"] == "time_point":
        max_score = 0
        for time_point in timeinfo["time"]:
            # 将时间戳转换为datetime对象
            dt1 = datetime.strptime(time_point, "%Y-%m-%d %H:%M:%S")
            dt2 = datetime.fromtimestamp(timeStamp)
            # 计算两个datetime对象之间的时间间隔
            delta = dt2 - dt1
            # 检查时间间隔是否不超过一天
            if delta <= timedelta(days=1):
                return 1
            else:
                max_score = max(max_score, 1 - abs(dt1.timestamp() - timeStamp) / pow(10, 8))
        return max_score
    else:
        return 1 - (int(time.time()) - timeStamp) / pow(10, 8)


def cal_domain_score(domain):
    score_map = {"jhsjk.people.cn": 0.8, "baike.baidu.com": 1}
    return score_map[domain]


modelPool = {}


def load_rerank_model(model: str, device: str):
    if str not in modelPool.keys():
        modelPool[str] = ReRankModel(model, device)
    return modelPool[str]


if __name__ == "__main__":
    # pairs = [['what is panda?', 'hi'], ['what is panda?',
    #                                     'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']]
    # print(ReRankModel().rerank(pairs, 2))
    print(cal_time_score(1619712000000, "最近一个月习近平去了哪里"))
