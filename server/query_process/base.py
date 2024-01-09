import json
import logging
import os

from configs import LLM_MODEL, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, LOG_PATH
from server.query_process.query_analysis import query_ner
import jionlp as jio
import time
from datetime import datetime


def get_logger(name: str):
    logger = logging.getLogger(name)
    filename = f'{datetime.now().date()}_{name}.log'
    has_file_handler = False
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            has_file_handler = True
            handler.baseFilename = os.path.join(LOG_PATH, filename)
            break
    if not has_file_handler:
        # 创建一个handler，用于写入日志文件
        fh = logging.FileHandler(os.path.join(LOG_PATH, filename), encoding='utf-8')
        # handler = TimedRotatingFileHandler(f'{name}.log', when='D', interval=1, backupCount=1000)
        # 再创建一个handler用于输出到控制台
        # ch = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    # 定义控制台输出层级
    # logger.setLevel(logging.DEBUG)
    # 为文件操作符绑定格式（可以绑定多种格式例fh.setFormatter(formatter2)）
    # fh.setFormatter(formatter)
    # 为控制台操作符绑定格式（可以绑定多种格式例ch.setFormatter(formatter2)）
    # ch.setFormatter(formatter)
    # 给logger对象绑定文件操作符
    # logger.addHandler(fh)
    # 给logger对象绑定文件操作符
    # logger.addHandler(ch)
    return logger


logger = get_logger("chat")


def query_time_extract(query: str):
    entity = query_ner(query)
    logger.info(f"query_ner:{entity}")
    time_dict = time_info()
    time_words = []
    for item in entity:
        if item["entity"] == "DATE":
            isUse = False
            for key, value in time_dict.items():
                if item["words"] in value:
                    isUse = True
                    time_words.append(jio.parse_time(key.split("> ")[1].strip(), time_base=time.time()))
                    query = query.replace(item["words"], "")
                    break
            if not isUse:
                time_change = jio.parse_time(item["words"], time_base=time.time())
                try:
                    if time_change["type"] == "time_span" or time_change["type"] == "time_point":
                        time_words.append(time_change)
                        query = query.replace(item["words"], "")
                except:
                    pass
    return query, time_words


def query_time_process(query: str):
    entity = query_ner(query)
    logger.info(f"query_ner:{entity}")
    time_dict = time_info()
    # ner识别出时间实体，替换为可解析时间
    for item in entity:
        if item["entity"] == "DATE":
            for key, value in time_dict.items():
                if item["words"] in value:
                    query = query.replace(item["words"], key.split("> ")[1].strip())
                    break
    parse_info = jio.ner.extract_time(query, time_base=time.time(), with_parsing=True)
    # print(parse_info)
    for item in parse_info:
        # 替换指定索引位置的字符
        start_index, end_index = item["offset"][0], item["offset"][1]
        if item["detail"]["type"] == "time_span":
            start_time = datetime.strptime(item["detail"]["time"][0].split(" ")[0], "%Y-%m-%d").strftime("%Y年%m月%d日")
            end_time = datetime.strptime(item["detail"]["time"][1].split(" ")[0], "%Y-%m-%d").strftime("%Y年%m月%d日")
            query = query[:start_index] + start_time + "到" + end_time + query[end_index:]
        elif item["detail"]["type"] == "time_point":
            time_set = set(datetime.strptime(time.split(" ")[0], "%Y-%m-%d").strftime("%Y年%m月%d日") for time in
                           item["detail"]["time"])
            query = query[:start_index] + "在" + ",".join(time_set) + "这天" + query[end_index + 1:]
        continue  # 只处理第一个时间
    # print(query)
    return query


def time_info():
    time_dict = {}
    with open("/data/lrq/llm/sync/Langchain-Chatchat/server/query_process/timewords.txt", "r") as f:
        time_info = f.readlines()
        time_words = []
        index = None
        for info in time_info:
            info = info.strip()
            if "<" in info and ">" in info:
                if index:
                    time_dict[index] = time_words
                    time_words = []
                    index = None
                index = info
            elif info:
                time_words.append(info)
        if index:
            time_dict[index] = time_words
    # print(time_dict)
    return time_dict


if __name__ == "__main__":
    query = "习近平这个月12号，13号，15号干了什么事"
    print(query_time_extract(query))
    # print(datetime.strptime("2012-01-01 00:00:00", "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d"))
