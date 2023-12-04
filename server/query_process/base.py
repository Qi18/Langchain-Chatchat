import json
import logging
import os
from datetime import datetime
from typing import List

from configs import LLM_MODEL, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, LOG_PATH
from server.chat.chat import chatWithHistory, chatOnes
from server.chat.utils import History
from server.knowledge_base.kb_doc_api import search_docs_multiQ, search_docs


def get_logger(name: str):
    logger = logging.getLogger(name)
    # 创建一个handler，用于写入日志文件
    filename = f'{datetime.now().date()}_{name}.log'
    fh = logging.FileHandler(os.path.join(LOG_PATH, filename), mode='w+', encoding='utf-8')
    # 再创建一个handler用于输出到控制台
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
    logger.setLevel(logging.INFO)
    # 定义控制台输出层级
    # logger.setLevel(logging.DEBUG)
    # 为文件操作符绑定格式（可以绑定多种格式例fh.setFormatter(formatter2)）
    fh.setFormatter(formatter)
    # 为控制台操作符绑定格式（可以绑定多种格式例ch.setFormatter(formatter2)）
    ch.setFormatter(formatter)
    # 给logger对象绑定文件操作符
    logger.addHandler(fh)
    # 给logger对象绑定文件操作符
    logger.addHandler(ch)
    return logger


logger = get_logger("chat")


def enhance_query_search(query: str,
                         knowledge_base_name: str,
                         history: List[History],
                         model_name: str = LLM_MODEL,
                         top_k: int = VECTOR_SEARCH_TOP_K,
                         score_threshold: float = SCORE_THRESHOLD,
                         history_query: bool = False,
                         multi_query: bool = False):
    logger.info(f"用户输入：{query}")
    # 优化query中的时间信息
    query = query_time_process(query)
    # 通过历史对话优化query
    if len(history) == 0:
        history_query = False
    retry = 3
    if history_query:
        new_query = historyQuery(query=query, history=history, model_name=model_name)
        while retry > 0:
            try:
                query = json.loads(new_query)["question"]
                retry = -2
            except ValueError:
                retry -= 1
                new_query = historyQuery(query=query, history=history)

    # 通过先验知识优化query
    retry = 3
    if multi_query:
        new_querys = multiQuery(query=query, history=history)
        while retry > 0:
            try:
                json.loads(new_querys)
                retry = -2
            except ValueError:
                retry -= 1
                new_querys = multiQuery(query=query, history=history)

    if retry == -2:
        print(f"multiquery:{new_querys}")
        multiquery = [query]
        multiquery.extend([item["query"] for item in json.loads(new_querys)])
        docs = search_docs_multiQ(querys=multiquery, knowledge_base_name=knowledge_base_name, top_k=top_k,
                                  score_threshold=score_threshold, search_method="hybrid")
    else:
        # logger = build_logger("chat", f"{datetime.now().date()}_chat.log")
        logger.info(f"最后查询：{query}")
        docs = search_docs(query, knowledge_base_name, top_k, score_threshold, search_method="hybrid")
    return docs


def multiQuery(query: str,
               history: [History],
               model_name: str = LLM_MODEL, ):
    from datetime import datetime
    current_time = datetime.today()
    info = {"question": query, "info": f"现在的时间是{current_time}", "num": 3}
    return chatWithHistory(info, "pre_chat", history, model_name=model_name, temperature=0.1)


def historyQuery(query: str,
                 history: [History],
                 model_name: str = LLM_MODEL, ):
    info = {"question": query, "chat_history": [i.to_msg_template() for i in history]}
    return chatOnes(info, "history_enQuery", model_name=model_name, temperature=0.1)


def query_time_process(query: str):
    import jionlp as jio
    import time
    from datetime import datetime
    # filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "tool", "timewords")
    # with open(filepath, "r", encoding="utf-8") as f:
    #     time_words = f.readlines()
    #     for time_word in time_words:
    #         if time_word.strip() in query:
    #             query = query.replace(time_word.strip(), "近一个月")
    #             break
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

if __name__ =="__main__":
    query = "2023年12月04日习近平去了哪里"
    print(query_time_process(query))
