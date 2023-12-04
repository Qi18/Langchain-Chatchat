import json

from fastapi import Body
from fastapi.responses import StreamingResponse
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from configs import LLM_MODEL, TEMPERATURE
from server.utils import wrap_done, get_ChatOpenAI, fschat_openai_api_address, get_model_worker_config
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from typing import List
from server.chat.utils import History
from server.utils import get_prompt_template


async def chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
               history: List[History] = Body([],
                                             description="历史对话",
                                             examples=[[
                                                 {"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                 {"role": "assistant", "content": "虎头虎脑"}]]
                                             ),
               stream: bool = Body(False, description="流式输出"),
               model_name: str = Body(LLM_MODEL, description="LLM 模型名称。"),
               temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
               # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
               prompt_name: str = Body("llm_chat",
                                       description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
               ):
    history = [History.from_data(h) for h in history]

    async def chat_iterator(query: str,
                            history: List[History] = [],
                            model_name: str = LLM_MODEL,
                            prompt_name: str = prompt_name,
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
                yield token
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield answer

        await task

    return StreamingResponse(chat_iterator(query=query,
                                           history=history,
                                           model_name=model_name,
                                           prompt_name=prompt_name),
                             media_type="text/event-stream")


def chat_local(query: str,
               history: List[History] = [],
               model_name: str = LLM_MODEL,
               temperature: float = TEMPERATURE,
               chunk_size: int = 1000,
               chunk_overlap: int = 0
               ) -> str:
    history = [History.from_data(h) for h in history]
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    config = get_model_worker_config(model_name)
    model = ChatOpenAI(
        verbose=True,
        openai_api_key=config.get("api_key", "EMPTY"),
        openai_api_base=config.get("api_base_url", fschat_openai_api_address()),
        model_name=model_name,
        temperature=temperature,
        openai_proxy=config.get("openai_proxy")
    )
    prompt_template = get_prompt_template("llm_chat")
    input_msg = History(role="user", content=prompt_template).to_msg_template(False)
    chat_prompt = ChatPromptTemplate.from_messages(
        [i.to_msg_template() for i in history] + [input_msg])
    chain = LLMChain(prompt=chat_prompt, llm=model)
    return chain.run({"input": query})


def chatWithHistory(info: json,
                    prompt_name: str,
                    history: [History],
                    model_name: str = LLM_MODEL,
                    temperature: float = TEMPERATURE, ):
    history = [History.from_data(h) for h in history]
    prompt_template = get_prompt_template(prompt_name)
    input_msg = History(role="user", content=prompt_template).to_msg_template(False)
    chat_prompt = ChatPromptTemplate.from_messages(
        [i.to_msg_template() for i in history] + [input_msg])
    config = get_model_worker_config(model_name)
    model = ChatOpenAI(
        verbose=True,
        openai_api_key=config.get("api_key", "EMPTY"),
        openai_api_base=config.get("api_base_url", fschat_openai_api_address()),
        model_name=model_name,
        temperature=temperature,
        openai_proxy=config.get("openai_proxy")
    )
    chain = LLMChain(prompt=chat_prompt, llm=model)
    return chain.run(info)


def chatOnes(info: json,
             prompt_name: str,
             model_name: str = LLM_MODEL,
             temperature: float = TEMPERATURE, ):
    prompt_template = get_prompt_template(prompt_name)
    input_msg = History(role="user", content=prompt_template).to_msg_template(False)
    chat_prompt = ChatPromptTemplate.from_messages([input_msg])
    config = get_model_worker_config(model_name)
    model = ChatOpenAI(
        verbose=True,
        openai_api_key=config.get("api_key", "EMPTY"),
        openai_api_base=config.get("api_base_url", fschat_openai_api_address()),
        model_name=model_name,
        temperature=temperature,
        openai_proxy=config.get("openai_proxy")
    )
    chain = LLMChain(prompt=chat_prompt, llm=model)
    return chain.run(info)


def chatRefine(query: str,
               model_name: str = LLM_MODEL,
               temperature: float = TEMPERATURE,
               chunk_size: int = 1000,
               chunk_overlap: int = 200):
    from text_splitter import ChineseRecursiveTextSplitter
    text_splitter = ChineseRecursiveTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_query = text_splitter.split_text(query)
    docs = [Document(page_content=t) for t in split_query]

    prompt_template = """总结下文内容:

        {text}

        总结内容:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    config = get_model_worker_config(model_name)
    model = ChatOpenAI(
        verbose=True,
        openai_api_key=config.get("api_key", "EMPTY"),
        openai_api_base=config.get("api_base_url", fschat_openai_api_address()),
        model_name=model_name,
        temperature=temperature,
        openai_proxy=config.get("openai_proxy")
    )
    # 定义refine合并总结内容的提示词模板
    refine_template = (
        "你的工作是负责生成一个最终的文本摘要\n"
        "这是现有的摘要信息: {existing_answer}\n"
        "根据新的背景信息完善现有的摘要"
        "背景信息如下\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "根据背景信息，完善现有的摘要"
        "如果背景信息没有用，则返回现有的摘要信息。"
    )
    refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )
    prompt_template = """将下文内容简化概括为大概2000字的简介:

           {text}

           简介:"""
    COM_PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
    from langchain.chains.summarize import load_summarize_chain
    # chain = load_summarize_chain(model, chain_type="map_reduce", return_intermediate_steps=True, verbose=True,
    #                              map_prompt=PROMPT, combine_prompt=COM_PROMPT, output_key="output_text",)
    chain = load_summarize_chain(model, chain_type="refine", return_intermediate_steps=True, verbose=True,
                                 question_prompt=PROMPT, refine_prompt=refine_prompt, output_key="output_text", )
    # 执行摘要任务
    result = chain({"input_documents": docs}, return_only_outputs=True)
    # print(result["output_text"])
    print("\n".join(result["intermediate_steps"]))
    return "\n".join(result["intermediate_steps"])


if __name__ == "__main__":
    pass
    # summary_local(query="你好,介绍一下清华大学")
    # summaryWithLLM(doc
