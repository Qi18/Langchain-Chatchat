import json
from typing import List

from configs import LLM_MODEL, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD
from server.chat.chat import chatWithHistory, chatOnes
from server.chat.utils import History
from server.knowledge_base.kb_doc_api import search_docs_multiQ, search_docs
from server.query_process.base import get_logger, query_time_process


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


def enhance_query_search(query: str,
                         knowledge_base_name: str,
                         history: List[History],
                         model_name: str = LLM_MODEL,
                         top_k: int = VECTOR_SEARCH_TOP_K,
                         score_threshold: float = SCORE_THRESHOLD,
                         history_query: bool = False,
                         multi_query: bool = False):
    logger = get_logger("chat")
    logger.info(f"用户输入：{query}")
    # 优化query中的时间信息
    # query = query_time_process(query)
    # logger.info(f"时间优化后的query：{query}")
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
