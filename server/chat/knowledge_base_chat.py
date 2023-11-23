import logging
from datetime import datetime
import time

from fastapi import Body, Request
from fastapi.responses import StreamingResponse
from fastchat.utils import build_logger
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import BasePromptTemplate, Document

from configs import (LLM_MODEL, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, TEMPERATURE, LOG_PATH)
from server.chat.chat import chatWithHistory, chatOnes
from server.utils import wrap_done, get_ChatOpenAI, get_model_worker_config, fschat_openai_api_address
from server.utils import BaseResponse, get_prompt_template
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBService, KBServiceFactory
from server.knowledge_base.kb_doc_api import search_docs, search_docs_multiQ
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
import json
import os
from urllib.parse import urlencode


async def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                              knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                              top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                              score_threshold: float = Body(SCORE_THRESHOLD,
                                                            description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                                                            ge=0, le=2),
                              history: List[History] = Body([],
                                                            description="历史对话",
                                                            examples=[[
                                                                {"role": "user",
                                                                 "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                                {"role": "assistant",
                                                                 "content": "虎头虎脑"}]]
                                                            ),
                              stream: bool = Body(False, description="流式输出"),
                              model_name: str = Body(LLM_MODEL, description="LLM 模型名称。"),
                              temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                              prompt_name: str = Body("knowledge_base_chat",
                                                      description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                              local_doc_url: bool = Body(False, description="知识文件返回本地路径(true)或URL(false)"),
                              request: Request = None,
                              ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    history = [History.from_data(h) for h in history]

    async def knowledge_base_chat_iterator(query: str,
                                           top_k: int,
                                           history: Optional[List[History]],
                                           model_name: str = LLM_MODEL,
                                           prompt_name: str = prompt_name,
                                           ) -> AsyncIterable[str]:
        logger = get_logger("chat")
        start_time = time.time()
        callback = AsyncIteratorCallbackHandler()
        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            callbacks=[callback],
        )

        # 通过历史对话优化query
        history_query = False
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
        multi_query = False
        retry = 3
        if multi_query:
            new_querys = enhanceQuery(query=query, history=history)
            while retry > 0:
                try:
                    json.loads(new_querys)
                    retry = -2
                except ValueError:
                    retry -= 1
                    new_querys = enhanceQuery(query=query, history=history)

        if retry == -2:
            print(f"multiquery:{new_querys}")
            multiquery = [query]
            multiquery.extend([item["query"] for item in json.loads(new_querys)])
            docs = search_docs_multiQ(querys=multiquery, knowledge_base_name=knowledge_base_name, top_k=top_k,
                                      score_threshold=score_threshold, search_method="hybrid")
        else:
            # logger = build_logger("chat", f"{datetime.now().date()}_chat.log")
            logger.info(f"用户输入：{query}")
            docs = search_docs(query, knowledge_base_name, top_k, score_threshold, search_method="hybrid")

        if docs == None or len(docs) == 0:
            yield json.dumps({"answer": "抱歉，在知识库中没有找到相关的信息。请您尝试更详细的阐述您的问题或者询问其他的事项，我会尽量为您解答！"}, ensure_ascii=False)
            return

        context = "\n---\n".join(
            [os.path.basename(doc.metadata.get("source")) + "\n" + doc.page_content for doc in docs])
        # print("context:\n" + context)
        # print("history:\n")
        # for i in history:
        #     print(i.content)
        prompt_template = get_prompt_template(prompt_name)
        input_msg = History(role="user", content=prompt_template).to_msg_template(False)

        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

        source_documents = []
        for inum, doc in enumerate(docs):
            filename = os.path.split(doc.metadata["source"])[-1]
            if local_doc_url:
                url = "file://" + doc.metadata["source"]
            else:
                parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                url = f"{request.base_url}knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": source_documents},
                             ensure_ascii=False)

        await task
        logger.info(f"chat响应时间: {time.time() - start_time}")


    return StreamingResponse(knowledge_base_chat_iterator(query=query,
                                                          top_k=top_k,
                                                          history=history,
                                                          model_name=model_name,
                                                          prompt_name=prompt_name),
                             media_type="text/event-stream")


def enhanceQuery(query: str,
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


if __name__ == "__main__":
    print(historyQuery("会议精神", [History(role='user', content='二十大'), History(role='assistant',
                                                                                    content='根据已知信息，中国共产党第二十次全国代表大会将于10月16日在北京召开。这次大会是在全党全国各族人民迈上全面建设社会主义现代化国家新征程、向第二个百年奋斗目标进军的关键时刻召开的一次十分重要的大会。大会的主要议题包括认真总结过去5年工作，全面总结新时代以来以习近平同志为核心的党中央团结带领全党全国各族人民坚持和发展中国特色社会主义取得的重大成就和宝贵经验，制定行动纲领和大政方针，动员全党全国各族人民坚定历史自信、增强历史主动，守正创新、勇毅前行，继续统筹推进“五位一体”总体布局、协调推进“四个全面”战略布局，继续扎实推进全体人民共同富裕，继续有力推进党的建设新的伟大工程，继续积极推动构建人类命运共同体。大会将选举产生新一届中央委员会和中央纪律检查委员会。')]))
