from langchain.chat_models import ChatOpenAI
from langchain.retrievers import MultiQueryRetriever

from configs.model_config import LLM_MODEL
from langchain import LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)

from server.chat.utils import History
from server.utils import get_model_worker_config, get_prompt_template, fschat_openai_api_address
from langchain.vectorstores import elasticsearch


from langchain.llms import OpenAI
from langchain.docstore import Wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer

docstore = DocstoreExplorer(Wikipedia())
tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description="useful for when you need to ask with search",
    ),
    Tool(
        name="Lookup",
        func=docstore.lookup,
        description="useful for when you need to ask with lookup",
    ),
]
model_name = LLM_MODEL
config = get_model_worker_config(model_name)
llm = ChatOpenAI(
    verbose=True,
    openai_api_key=config.get("api_key", "EMPTY"),
    openai_api_base=config.get("api_base_url", fschat_openai_api_address()),
    model_name=model_name,
    temperature=0.1,
    openai_proxy=config.get("openai_proxy")
)
react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=True)
question = "介绍一下习近平和李克强"
react.run(question)
