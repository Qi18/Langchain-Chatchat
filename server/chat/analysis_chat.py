from server.query_process.query_analysis import query_ir
from fastapi import Body, Request
from fastapi.responses import StreamingResponse

from configs import (LLM_MODEL, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, TEMPERATURE, LOG_PATH)
from server.query_process.base import enhance_query_search, logger
from server.utils import wrap_done, get_ChatOpenAI, get_model_worker_config, fschat_openai_api_address
from server.utils import BaseResponse, get_prompt_template
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBService, KBServiceFactory
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
import json
import os
from urllib.parse import urlencode


async def analysis_chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
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
                                                          {"role": "assistant", "content": "虎头虎脑"}]]
                                                      ),
                        stream: bool = Body(False, description="流式输出"),
                        model_name: str = Body(LLM_MODEL, description="LLM 模型名称。"),
                        temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                        local_doc_url: bool = Body(False, description="知识文件返回本地路径(true)或URL(false)"),
                        request: Request = None,
                        ):
    history = [History.from_data(h) for h in history]
    intent = query_ir(query)
    logger.info(f"query intent判别：{intent}")
    if intent == "query" or intent == "hint_query":
        async def knowledge_base_chat_iterator(query: str,
                                               top_k: int,
                                               history: Optional[List[History]],
                                               model_name: str = LLM_MODEL,
                                               prompt_name: str = "knowledge_base_chat",
                                               ) -> AsyncIterable[str]:
            # import time
            # start_time = time.time()
            callback = AsyncIteratorCallbackHandler()
            model = get_ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                callbacks=[callback],
            )

            # 优化query和搜索部分
            docs = enhance_query_search(query=query, knowledge_base_name=knowledge_base_name, top_k=top_k,
                                        score_threshold=score_threshold, model_name=model_name, history=history)

            if docs == None or len(docs) == 0:
                yield json.dumps({
                    "answer": "抱歉，在知识库中没有找到相关的信息。请您尝试更详细的阐述您的问题或者询问其他的事项，我会尽量为您解答！"},
                    ensure_ascii=False)
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
                # time = datetime.fromtimestamp(int(doc.metadata["publishTime"]) / 1000).strftime("%Y-%m-%d")
                text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content} \n\n"""
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

        return StreamingResponse(knowledge_base_chat_iterator(query=query,
                                                              top_k=top_k,
                                                              history=history,
                                                              model_name=model_name,),
                                 media_type="text/event-stream")
    # elif intent == "chat" or intent == "hint_chat":
    else:
        async def chat_iterator(query: str,
                                history: List[History] = [],
                                model_name: str = LLM_MODEL,
                                prompt_name: str = "llm_chat",
                                ) -> AsyncIterable[str]:
            callback = AsyncIteratorCallbackHandler()
            model = get_ChatOpenAI(
                model_name=model_name,
                temperature=temperature,
                callbacks=[callback],
            )

            prompt_template = get_prompt_template(prompt_name)
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_template() for i in history] + [input_msg])
            chain = LLMChain(prompt=chat_prompt, llm=model)

            # Begin a task that runs in the background.
            task = asyncio.create_task(wrap_done(
                chain.acall({"input": query}),
                callback.done),
            )

            if stream:
                async for token in callback.aiter():
                    # Use server-sent-events to stream the response
                    yield json.dumps({"answer": token}, ensure_ascii=False)
            else:
                answer = ""
                async for token in callback.aiter():
                    answer += token
                yield json.dumps({"answer": answer}, ensure_ascii=False)

            await task

        return StreamingResponse(chat_iterator(query=query,
                                               history=history,
                                               model_name=model_name,),
                                 media_type="text/event-stream")
