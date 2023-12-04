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
        import time
        start_time = time.time()
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
        # logger.info(f"chat响应时间: {time.time() - start_time}")

    return StreamingResponse(knowledge_base_chat_iterator(query=query,
                                                          top_k=top_k,
                                                          history=history,
                                                          model_name=model_name,
                                                          prompt_name=prompt_name),
                             media_type="text/event-stream")


if __name__ == "__main__":
    pass
    # query = "习近平最近几个月有没有参加重要峰会？"
    # filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tool", "timewords")
    # with open(filepath, "r", encoding="utf-8") as f:
    #     time_words = f.readlines()
    #     for time_word in time_words:
    #         if time_word.strip() in query:
    #             query = query.replace(time_word.strip(), "近一个月")
    #             break
    # parse_info = jio.ner.extract_time(query, time_base=time.time(), with_parsing=True)
    # print(parse_info)
    # for item in parse_info:
    #     # 替换指定索引位置的字符
    #     start_index, end_index = item["offset"][0], item["offset"][1]
    #     if item["detail"]["type"] == "time_span":
    #         date_str = "2012-02-09"
    #         start_time = datetime.strptime(item["detail"]["time"][0].split(" ")[0], "%Y-%m-%d").strftime("%Y年%m月%d日")
    #         end_time = datetime.strptime(item["detail"]["time"][1].split(" ")[0], "%Y-%m-%d").strftime("%Y年%m月%d日")
    #         query = query[:start_index] + start_time + "到" + end_time+ query[end_index:]
    #     elif item["detail"]["type"] == "time_point":
    #         time_set = set(datetime.strptime(time.split(" ")[0], "%Y-%m-%d").strftime("%Y年%m月%d日") for time in item["detail"]["time"])
    #         query = query[:start_index] + "在" + ",".join(time_set) + "这天" + query[end_index + 1:]
    #     continue # 只处理第一个时间
    # print(query)