# -*- coding: utf-8 -*-
import os
from datetime import datetime

import jionlp as jio

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from configs import (
    EMBEDDING_MODEL,
    KB_ROOT_PATH,
    CHUNK_SIZE,
    OVERLAP_SIZE,
    ZH_TITLE_ENHANCE,
    logger,
    log_verbose,
    text_splitter_dict,
    LLM_MODEL,
    TEXT_SPLITTER_NAME,
    VLLM_MODEL_DICT, RERANK_MODEL,
)
import importlib
from text_splitter import zh_title_enhance as func_zh_title_enhance, ChineseRecursiveTextSplitter, LGTextSplitter
import langchain.document_loaders
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
from server.utils import run_in_thread_pool, embedding_device, get_model_worker_config, rerank_device
import io
from typing import List, Union, Callable, Dict, Optional, Tuple, Generator
import chardet
from rich import print


def validate_kb_name(knowledge_base_id: str) -> bool:
    # 检查是否包含预期外的字符或路径攻击关键字
    if "../" in knowledge_base_id:
        return False
    return True


def get_kb_path(knowledge_base_name: str):
    return os.path.join(KB_ROOT_PATH, knowledge_base_name)


def get_doc_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "content")


def get_vs_path(knowledge_base_name: str, vector_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), vector_name)


def get_file_path(knowledge_base_name: str, doc_name: str):
    return os.path.join(get_doc_path(knowledge_base_name), doc_name)


def list_kbs_from_folder():
    return [f for f in os.listdir(KB_ROOT_PATH)
            if os.path.isdir(os.path.join(KB_ROOT_PATH, f))]


def list_files_from_folder(kb_name: str):
    doc_path = get_doc_path(kb_name)
    return [file for file in os.listdir(doc_path)
            if os.path.isfile(os.path.join(doc_path, file))]


def load_embeddings(model: str = EMBEDDING_MODEL, device: str = embedding_device()):
    '''
    从缓存中加载embeddings，可以避免多线程时竞争加载。
    '''
    from server.knowledge_base.kb_cache.base import embeddings_pool
    # if "bge-" in model:
    #     return embeddings_pool.load_bge_embeddings(model=model, device=device)
    return embeddings_pool.load_embeddings(model=model, device=device)


LOADER_DICT = {"UnstructuredHTMLLoader": ['.html'],
               "UnstructuredMarkdownLoader": ['.md'],
               "CustomJSONLoader": [".json"],
               "CSVLoader": [".csv"],
               "RapidOCRPDFLoader": [".pdf"],
               "RapidOCRLoader": ['.png', '.jpg', '.jpeg', '.bmp'],
               "UnstructuredFileLoader": ['.eml', '.msg', '.rst',
                                          '.rtf', '.txt', '.xml',
                                          '.docx', '.epub', '.odt',
                                          '.ppt', '.pptx', '.tsv'],
               }
SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]


class CustomJSONLoader(langchain.document_loaders.JSONLoader):
    '''
    langchain的JSONLoader需要jq，在win上使用不便，进行替代。针对langchain==0.0.286
    '''

    def __init__(
            self,
            file_path: Union[str, Path],
            content_key: Optional[str] = None,
            metadata_func: Optional[Callable[[Dict, Dict], Dict]] = None,
            text_content: bool = True,
            json_lines: bool = False,
    ):
        """Initialize the JSONLoader.

        Args:
            file_path (Union[str, Path]): The path to the JSON or JSON Lines file.
            content_key (str): The key to use to extract the content from the JSON if
                results to a list of objects (dict).
            metadata_func (Callable[Dict, Dict]): A function that takes in the JSON
                object extracted by the jq_schema and the default metadata and returns
                a dict of the updated metadata.
            text_content (bool): Boolean flag to indicate whether the content is in
                string format, default to True.
            json_lines (bool): Boolean flag to indicate whether the input is in
                JSON Lines format.
        """
        self.file_path = Path(file_path).resolve()
        self._content_key = content_key
        self._metadata_func = metadata_func
        self._text_content = text_content
        self._json_lines = json_lines

    def _parse(self, content: str, docs: List[Document]) -> None:
        """Convert given content to documents."""
        data = json.loads(content)

        # Perform some validation
        # This is not a perfect validation, but it should catch most cases
        # and prevent the user from getting a cryptic error later on.
        if self._content_key is not None:
            self._validate_content_key(data)
        if self._metadata_func is not None:
            self._validate_metadata_func(data)

        for i, sample in enumerate(data, len(docs) + 1):
            text = self._get_text(sample=sample)
            metadata = self._get_metadata(
                sample=sample, source=str(self.file_path), seq_num=i
            )
            docs.append(Document(page_content=text, metadata=metadata))


langchain.document_loaders.CustomJSONLoader = CustomJSONLoader


def get_LoaderClass(file_extension):
    for LoaderClass, extensions in LOADER_DICT.items():
        if file_extension in extensions:
            return LoaderClass


# 把一些向量化共用逻辑从KnowledgeFile抽取出来，等langchain支持内存文件的时候，可以将非磁盘文件向量化
def get_loader(loader_name: str, file_path_or_content: Union[str, bytes, io.StringIO, io.BytesIO]):
    '''
    根据loader_name和文件路径或内容返回文档加载器。
    '''
    try:
        if loader_name in ["RapidOCRPDFLoader", "RapidOCRLoader"]:
            document_loaders_module = importlib.import_module('document_loaders')
        else:
            document_loaders_module = importlib.import_module('langchain.document_loaders')
        DocumentLoader = getattr(document_loaders_module, loader_name)
    except Exception as e:
        msg = f"为文件{file_path_or_content}查找加载器{loader_name}时出错：{e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        document_loaders_module = importlib.import_module('langchain.document_loaders')
        DocumentLoader = getattr(document_loaders_module, "UnstructuredFileLoader")

    if loader_name == "UnstructuredFileLoader":
        loader = DocumentLoader(file_path_or_content, autodetect_encoding=True)
    elif loader_name == "CSVLoader":
        # 自动识别文件编码类型，避免langchain loader 加载文件报编码错误
        with open(file_path_or_content, 'rb') as struct_file:
            encode_detect = chardet.detect(struct_file.read())
        if encode_detect:
            loader = DocumentLoader(file_path_or_content, encoding=encode_detect["encoding"])
        else:
            loader = DocumentLoader(file_path_or_content, encoding="utf-8")

    elif loader_name == "JSONLoader":
        loader = DocumentLoader(file_path_or_content, jq_schema=".", text_content=False)
    elif loader_name == "CustomJSONLoader":
        loader = DocumentLoader(file_path_or_content, text_content=False)
    elif loader_name == "UnstructuredMarkdownLoader":
        loader = DocumentLoader(file_path_or_content, mode="elements")
    elif loader_name == "UnstructuredHTMLLoader":
        loader = DocumentLoader(file_path_or_content, mode="elements")
    else:
        loader = DocumentLoader(file_path_or_content)
    return loader


def make_text_splitter(
        splitter_name: str = TEXT_SPLITTER_NAME,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = OVERLAP_SIZE,
        llm_model: str = LLM_MODEL,
):
    """
    根据参数获取特定的分词器
    """
    splitter_name = splitter_name or "SpacyTextSplitter"
    try:
        if splitter_name == "MarkdownHeaderTextSplitter":  # MarkdownHeaderTextSplitter特殊判定
            headers_to_split_on = text_splitter_dict[splitter_name]['headers_to_split_on']
            text_splitter = langchain.text_splitter.MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on)
        else:

            try:  ## 优先使用用户自定义的text_splitter
                text_splitter_module = importlib.import_module('text_splitter')
                TextSplitter = getattr(text_splitter_module, splitter_name)
            except:  ## 否则使用langchain的text_splitter
                text_splitter_module = importlib.import_module('langchain.text_splitter')
                TextSplitter = getattr(text_splitter_module, splitter_name)

            if text_splitter_dict[splitter_name]["source"] == "tiktoken":  ## 从tiktoken加载
                try:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                except:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
            elif text_splitter_dict[splitter_name]["source"] == "huggingface":  ## 从huggingface加载
                if text_splitter_dict[splitter_name]["tokenizer_name_or_path"] == "":
                    config = get_model_worker_config(llm_model)
                    text_splitter_dict[splitter_name]["tokenizer_name_or_path"] = \
                        config.get("model_path")

                if text_splitter_dict[splitter_name]["tokenizer_name_or_path"] == "gpt2":
                    from transformers import GPT2TokenizerFast
                    from langchain.text_splitter import CharacterTextSplitter
                    # tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                    tokenizer = GPT2TokenizerFast.from_pretrained(VLLM_MODEL_DICT["gpt2"])
                else:  ## 字符长度加载
                    tokenizer = AutoTokenizer.from_pretrained(
                        text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                        trust_remote_code=True)
                text_splitter = TextSplitter.from_huggingface_tokenizer(
                    tokenizer=tokenizer,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            else:
                try:
                    text_splitter = TextSplitter(
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                except:
                    text_splitter = TextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
    except Exception as e:
        print(e)
        text_splitter_module = importlib.import_module('langchain.text_splitter')
        TextSplitter = getattr(text_splitter_module, "RecursiveCharacterTextSplitter")
        text_splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter


class KnowledgeFile:
    def __init__(
            self,
            filename: str,
            knowledge_base_name: str,
            zh_title_enhance: bool = ZH_TITLE_ENHANCE,
            chunk_size: int = CHUNK_SIZE,
            chunk_overlap: int = OVERLAP_SIZE,
            metadata: Optional[Dict] = None,
    ):
        '''
        对应知识库目录中的文件，必须是磁盘上存在的才能进行向量化等操作。
        '''
        self.kb_name = knowledge_base_name
        self.filename = filename
        self.ext = os.path.splitext(filename)[-1].lower()
        if self.ext not in SUPPORTED_EXTS:
            raise ValueError(f"暂未支持的文件格式 {self.ext}")
        self.filepath = get_file_path(knowledge_base_name, filename)
        self.docs = None
        self.splited_docs = None
        self.full_docs = None
        self.document_loader_name = get_LoaderClass(self.ext)
        self.text_splitter_name = TEXT_SPLITTER_NAME
        self.zh_title_enhance = zh_title_enhance
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.metadata = metadata

    def file2docs(self, refresh: bool = False):
        try:
            if self.docs is None or refresh:
                logger.info(f"{self.document_loader_name} used for {self.filepath}")
                loader = get_loader(self.document_loader_name, self.filepath)
                self.docs = loader.load()
                timeInfo = []
                for doc in self.docs:
                    timeInfo.append(extractTimeInfo(doc.page_content))
                if timeInfo:
                    if self.metadata is None:
                        self.metadata = {"contentTime": timeInfo}
                    else:
                        self.metadata["contentTime"] = timeInfo
            if self.metadata:
                for doc in self.docs:
                    for key in self.metadata.keys():
                        doc.metadata[key] = self.metadata[key]
            return self.docs
        ##可能会有文件名过长的情况
        except Exception as e:
            print(e)
            return ""

    def docs2texts(
            self,
            docs: List[Document] = None,
            refresh: bool = False,
            text_splitter: TextSplitter = None,
    ):
        docs = docs or self.file2docs(refresh=refresh)
        if not docs:
            return []
        if self.ext not in [".csv"]:
            if text_splitter is None:
                text_splitter = make_text_splitter(splitter_name=self.text_splitter_name, chunk_size=self.chunk_size,
                                                   chunk_overlap=self.chunk_overlap)
            if self.text_splitter_name == "MarkdownHeaderTextSplitter":
                docs = text_splitter.split_text(docs[0].page_content)
                for doc in docs:
                    # 如果文档有元数据
                    if doc.metadata:
                        doc.metadata["source"] = os.path.basename(self.filepath)
            else:
                docs = text_splitter.split_documents(docs)

        # 增加切分信息
        for doc_id in range(len(docs)):
            docs[doc_id].metadata["chunk_index"] = doc_id
            # if "publishTime" in self.metadata.keys():
            #     docs[doc_id].page_content = "发布于" + timeStampToTime(self.metadata["publishTime"]) + "\n" + docs[doc_id].page_content
            # docs[doc_id].page_content = os.path.splitext(self.filename)[0].split("_")[0] + "\n" + docs[doc_id].page_content
            if "publishTime" in self.metadata.keys():
                docs[doc_id].page_content = docs[doc_id].page_content + "\n" + "发布时间" + timeStampToTime(
                    self.metadata["publishTime"])
            docs[doc_id].page_content = docs[doc_id].page_content + "\n" + \
                                        os.path.splitext(self.filename)[0].split("_")[0]
        print(f"文档切分示例：{docs[0]}")
        if self.zh_title_enhance:
            docs = func_zh_title_enhance(docs)
        self.splited_docs = docs
        return self.splited_docs

    def file2text(
            self,
            docs: List[Document] = None,
            refresh: bool = False,
            text_splitter: TextSplitter = None,
    ):
        if self.splited_docs is None or refresh:
            docs = self.file2docs()
            self.splited_docs = self.docs2texts(docs=docs,
                                                refresh=refresh,
                                                text_splitter=text_splitter)
        return self.splited_docs

    def file2full_text(
            self,
            docs: List[Document] = None,
            zh_title_enhance: bool = ZH_TITLE_ENHANCE,
            refresh: bool = False,
    ):
        if self.full_docs is None or refresh:
            docs = self.file2docs()
            if not docs:
                return ""
            self.full_docs = Document(page_content=''.join(i.page_content for i in docs))
            if self.zh_title_enhance:
                self.full_docs = func_zh_title_enhance(self.full_docs)
        return self.full_docs

    def file_exist(self):
        return os.path.isfile(self.filepath)

    def get_mtime(self):
        return os.path.getmtime(self.filepath)

    def get_size(self):
        return os.path.getsize(self.filepath)


def files2docs_in_thread(
        files: List[Union[KnowledgeFile, Tuple[str, str], Dict]],
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = OVERLAP_SIZE,
        zh_title_enhance: bool = ZH_TITLE_ENHANCE,
        pool: ThreadPoolExecutor = None,
) -> Generator:
    '''
    利用多线程批量将磁盘文件转化成langchain Document.
    如果传入参数是Tuple，形式为(filename, kb_name)
    生成器返回值为 status, (kb_name, file_name, docs | error)
    '''

    def file2docs(*, file: KnowledgeFile, **kwargs) -> Tuple[bool, Tuple[str, str, List[Document]]]:
        try:
            return True, (file.kb_name, file.filename, file.file2text(**kwargs))
        except Exception as e:
            msg = f"从文件 {file.kb_name}/{file.filename} 加载文档时出错：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            return False, (file.kb_name, file.filename, msg)

    kwargs_list = []
    for i, file in enumerate(files):
        kwargs = {}
        try:
            if isinstance(file, tuple) and len(file) >= 2:
                filename = file[0]
                kb_name = file[1]
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            elif isinstance(file, dict):
                filename = file.pop("filename")
                kb_name = file.pop("kb_name")
                kwargs.update(file)
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            kwargs["file"] = file
            kwargs["chunk_size"] = chunk_size
            kwargs["chunk_overlap"] = chunk_overlap
            kwargs["zh_title_enhance"] = zh_title_enhance
            kwargs_list.append(kwargs)
        except Exception as e:
            yield False, (kb_name, filename, str(e))

    for result in run_in_thread_pool(func=file2docs, params=kwargs_list, pool=pool):
        yield result


def timeStampToTime(timestamp):
    if len(str(timestamp)) == 13:
        timestamp = timestamp / 1000
    import datetime
    datetime = datetime.datetime.fromtimestamp(timestamp)
    return datetime.strftime('%Y年%m月%d日')


def extractTimeInfo(content: str):
    import time
    import re
    timeInfo = jio.ner.extract_time(content, time_base=time.time(), with_parsing=False)
    pattern = r'\d{4}年'
    time_list = []
    for item in timeInfo:
        matches = re.search(pattern, item["text"])
        if matches:
            time_list.append(item["text"])
    ans = []
    ans = set()
    for item in time_list:
        info = jio.parse_time(item)
        ans.add(datetime.strptime(info["time"][0], "%Y-%m-%d %H:%M:%S").timestamp())
        ans.add(datetime.strptime(info["time"][1], "%Y-%m-%d %H:%M:%S").timestamp())
    return [item for item in ans]
    # print(jio.)


if __name__ == "__main__":
    from pprint import pprint

    # kb_file = KnowledgeFile(filename="test.txt", knowledge_base_name="samples")
    # # kb_file.text_splitter_name = "RecursiveCharacterTextSplitter"
    # docs = kb_file.file2docs()
    # pprint(docs[-1])
    #
    # docs = kb_file.file2text()
    # pprint(docs[-1])
    # from transformers import GPT2TokenizerFast, AutoTokenizer

    # tokenizer = GPT2TokenizerFast.from_pretrained("../../tool/gpt2")
    # print(timeStampToTime(1609430400000))
    # filename = "新民主主义.txt"
    # loader = get_loader(os.path.splitext(filename)[-1].lower(),
    #                     "/data/lrq/llm/sync/Langchain-Chatchat/knowledge_base/习近平重要讲话数据库/content/新民主主义.txt")
    # docs = loader.load()
    # print(docs)
    # # text_splitter = ChineseRecursiveTextSplitter(
    # #     keep_separator=True,
    # #     is_separator_regex=True,
    # #     chunk_size=300,
    # #     chunk_overlap=90
    # # )
    # text_splitter = make_text_splitter(splitter_name=TEXT_SPLITTER_NAME, chunk_size=300,
    #                                    chunk_overlap=90)
    # chunks = text_splitter.split_documents(docs)
    # count = 0
    # for chunk in chunks:
    #     print(chunk)
    #     count += 1
    # print(count)
    print(extractTimeInfo(
        "中国共产党第十九届中央委员会第六次全体会议，于2021年11月8日至11日在北京举行。出席这次全会的有，中央委员197人，候补中央委员151人。中央纪律检查委员会常务委员会委员和有关方面负责同志列席会议。党的十九大代表中部分基层同志和专家学者也列席会议。全会由中央政治局主持。中央委员会总书记习近平作了重要讲话。中国共产党第十九届中央委员会第六次全体会议，于2021年11月8日至11日在北京举行。中央委员会总书记习近平作重要讲话。新华社记者 鞠鹏 摄 全会听取和讨论了习近平受中央政治局委托作的工作报告，审议通过了《中共中央关于党的百年奋斗重大成就和历史经验的决议》，审议通过了《关于召开党的第二十次全国代表大会的决议》。习近平就《中共中央关于党的百年奋斗重大成就和历史经验的决议（讨论稿）》向全会作了说明。全会充分肯定党的十九届五中全会以来中央政治局的工作。一致认为，一年来，世界百年未有之大变局和新冠肺炎疫情全球大流行交织影响，外部环境更趋复杂严峻，国内新冠肺炎疫情防控和经济社会发展各项任务极为繁重艰巨。中央政治局高举中国特色社会主义伟大旗帜，坚持以马克思列宁主义、毛泽东思想、邓小平理论、“三个代表”重要思想、科学发展观、习近平新时代中国特色社会主义思想为指导，全面贯彻党的十九大和十九届二中、三中、四中、五中全会精神，统筹国内国际两个大局，统筹疫情防控和经济社会发展，统筹发展和安全，坚持稳中求进工作总基调，全面贯彻新发展理念，加快构建新发展格局，经济保持较好发展态势，科技自立自强积极推进，改革开放不断深化，脱贫攻坚战如期打赢，民生保障有效改善，社会大局保持稳定，国防和军队现代化扎实推进，中国特色大国外交全面推进，党史学习教育扎实有效，战胜多种严重自然灾害，党和国家各项事业取得了新的重大成就。成功举办庆祝中国共产党成立100周年系列活动，中共中央总书记习近平发表重要讲话，正式宣布全面建成小康社会，激励全党全国各族人民意气风发踏上向第二个百年奋斗目标进军的新征程。全会认为，总结党的百年奋斗重大成就和历史经验，是在建党百年历史条件下开启全面建设社会主义现代化国家新征程、在新时代坚持和发展中国特色社会主义的需要；是增强政治意识、大局意识、核心意识、看齐意识，坚定道路自信、理论自信、制度自信、文化自信，做到坚决维护习近平同志党中央的核心、全党的核心地位，坚决维护党中央权威和集中统一领导，确保全党步调一致向前进的需要；是推进党的自我革命、提高全党斗争本领和应对风险挑战能力、永葆党的生机活力、团结带领全国各族人民为实现中华民族伟大复兴的中国梦而继续奋斗的需要。全党要坚持唯物史观和正确党史观，从党的百年奋斗中看清楚过去我们为什么能够成功、弄明白未来我们怎样才能继续成功，从而更加坚定、更加自觉地践行初心使命，在新时代更好坚持和发展中国特色社会主义。全会提出，中国共产党自一九二一年成立以来，始终把为中国人民谋幸福、为中华民族谋复兴作为自己的初心使命，始终坚持共产主义理想和社会主义信念，团结带领全国各族人民为争取民族独立、人民解放和实现国家富强、人民幸福而不懈奋斗，已经走过一百年光辉历程。党和人民百年奋斗，书写了中华民族几千年历史上最恢宏的史诗。全会提出，新民主主义革命时期，党面临的主要任务是，反对帝国主义、封建主义、官僚资本主义，争取民族独立、人民解放，为实现中华民族伟大复兴创造根本社会条件。在革命斗争中，以毛泽东同志为主要代表的中国共产党人，把马克思列宁主义基本原理同中国具体实际相结合，对经过艰苦探索、付出巨大牺牲积累的一系列独创性经验作了理论概括，开辟了农村包围城市、武装夺取政权的正确革命道路，创立了毛泽东思想，为夺取新民主主义革命胜利指明了正确方向。党领导人民浴血奋战、百折不挠，创造了新民主主义革命的伟大成就，成立中华人民共和国，实现民族独立、人民解放，彻底结束了旧中国半殖民地半封建社会的历史，彻底结束了极少数剥削者统治广大劳动人民的历史，彻底结束了旧中国一盘散沙的局面，彻底废除了列强强加给中国的不平等条约和帝国主义在中国的一切特权，实现了中国从几千年封建专制政治向人民民主的伟大飞跃，也极大改变了世界政治格局，鼓舞了全世界被压迫民族和被压迫人民争取解放的斗争。中国共产党和中国人民以英勇顽强的奋斗向世界庄严宣告，中国人民从此站起来了，中华民族任人宰割、饱受欺凌的时代一去不复返了，中国发展从此开启了新纪元。全会提出，社会主义革命和建设时期，党面临的主要任务是，实现从新民主主义到社会主义的转变，进行社会主义革命，推进社会主义建设，为实现中华民族伟大复兴奠定根本政治前提和制度基础。在这个时期，以毛泽东同志为主要代表的中国共产党人提出关于社会主义建设的一系列重要思想。毛泽东思想是马克思列宁主义在中国的创造性运用和发展，是被实践证明了的关于中国革命和建设的正确的理论原则和经验总结，是马克思主义中国化的第一次历史性飞跃。党领导人民自力更生、发愤图强，创造了社会主义革命和建设的伟大成就，实现了中华民族有史以来最为广泛而深刻的社会变革，实现了一穷二白、人口众多的东方大国大步迈进社会主义社会的伟大飞跃。我国建立起独立的比较完整的工业体系和国民经济体系，农业生产条件显著改变，教育、科学、文化、卫生、体育事业有很大发展，人民解放军得到壮大和提高，彻底结束了旧中国的屈辱外交。中国共产党和中国人民以英勇顽强的奋斗向世界庄严宣告，中国人民不但善于破坏一个旧世界、也善于建设一个新世界，只有社会主义才能救中国，只有社会主义才能发展中国。全会提出，改革开放和社会主义现代化建设新时期，党面临的主要任务是，继续探索中国建设社会主义的正确道路，解放和发展社会生产力，使人民摆脱贫困、尽快富裕起来，为实现中华民族伟大复兴提供充满新的活力的体制保证和快速发展的物质条件。党的十一届三中全会以后，以邓小平同志为主要代表的中国共产党人，团结带领全党全国各族人民，深刻总结新中国成立以来正反两方面经验，围绕什么是社会主义、怎样建设社会主义这一根本问题，借鉴世界社会主义历史经验，创立了邓小平理论，解放思想，实事求是，作出把党和国家工作中心转移到经济建设上来、实行改革开放的历史性决策，深刻揭示社会主义本质，确立社会主义初级阶段基本路线，明确提出走自己的路、建设中国特色社会主义，科学回答了建设中国特色社会主义的一系列基本问题，制定了到二十一世纪中叶分三步走、基本实现社会主义现代化的发展战略，成功开创了中国特色社会主义。全会提出，党的十三届四中全会以后，以江泽民同志为主要代表的中国共产党人，团结带领全党全国各族人民，坚持党的基本理论、基本路线，加深了对什么是社会主义、怎样建设社会主义和建设什么样的党、怎样建设党的认识，形成了“三个代表”重要思想，在国内外形势十分复杂、世界社会主义出现严重曲折的严峻考验面前捍卫了中国特色社会主义，确立了社会主义市场经济体制的改革目标和基本框架，确立了社会主义初级阶段公有制为主体、多种所有制经济共同发展的基本经济制度和按劳分配为主体、多种分配方式并存的分配制度，开创全面改革开放新局面，推进党的建设新的伟大工程，成功把中国特色社会主义推向二十一世纪。全会提出，党的十六大以后，以胡锦涛同志为主要代表的中国共产党人，团结带领全党全国各族人民，在全面建设小康社会进程中推进实践创新、理论创新、制度创新，深刻认识和回答了新形势下实现什么样的发展、怎样发展等重大问题，形成了科学发展观，抓住重要战略机遇期，聚精会神搞建设，一心一意谋发展，强调坚持以人为本、全面协调可持续发展，着力保障和改善民生，促进社会公平正义，推进党的执政能力建设和先进性建设，成功在新形势下坚持和发展了中国特色社会主义。全会强调，在这个时期，党从新的实践和时代特征出发坚持和发展马克思主义，科学回答了建设中国特色社会主义的发展道路、发展阶段、根本任务、发展动力、发展战略、政治保证、祖国统一、外交和国际战略、领导力量和依靠力量等一系列基本问题，形成中国特色社会主义理论体系，实现了马克思主义中国化新的飞跃。党领导人民解放思想、锐意进取，创造了改革开放和社会主义现代化建设的伟大成就，我国实现了从高度集中的计划经济体制到充满活力的社会主义市场经济体制、从封闭半封闭到全方位开放的历史性转变，实现了从生产力相对落后的状况到经济总量跃居世界第二的历史性突破，实现了人民生活从温饱不足到总体小康、奔向全面小康的历史性跨越，推进了中华民族从站起来到富起来的伟大飞跃。中国共产党和中国人民以英勇顽强的奋斗向世界庄严宣告，改革开放是决定当代中国前途命运的关键一招，中国特色社会主义道路是指引中国发展繁荣的正确道路，中国大踏步赶上了时代。全会提出，党的十八大以来，中国特色社会主义进入新时代。党面临的主要任务是，实现第一个百年奋斗目标，开启实现第二个百年奋斗目标新征程，朝着实现中华民族伟大复兴的宏伟目标继续前进。党领导人民自信自强、守正创新，创造了新时代中国特色社会主义的伟大成就。全会强调，以习近平同志为主要代表的中国共产党人，坚持把马克思主义基本原理同中国具体实际相结合、同中华优秀传统文化相结合，坚持毛泽东思想、邓小平理论、“三个代表”重要思想、科学发展观，深刻总结并充分运用党成立以来的历史经验，从新的实际出发，创立了习近平新时代中国特色社会主义思想。习近平同志对关系新时代党和国家事业发展的一系列重大理论和实践问题进行了深邃思考和科学判断，就新时代坚持和发展什么样的中国特色社会主义、怎样坚持和发展中国特色社会主义，建设什么样的社会主义现代化强国、怎样建设社会主义现代化强国，建设什么样的长期执政的马克思主义政党、怎样建设长期执政的马克思主义政党等重大时代课题，提出一系列原创性的治国理政新理念新思想新战略，是习近平新时代中国特色社会主义思想的主要创立者。习近平新时代中国特色社会主义思想是当代中国马克思主义、二十一世纪马克思主义，是中华文化和中国精神的时代精华，实现了马克思主义中国化新的飞跃。党确立习近平同志党中央的核心、全党的核心地位，确立习近平新时代中国特色社会主义思想的指导地位，反映了全党全军全国各族人民共同心愿，对新时代党和国家事业发展、对推进中华民族伟大复兴历史进程具有决定性意义。全会指出，以习近平同志为核心的党中央，以伟大的历史主动精神、巨大的政治勇气、强烈的责任担当，统筹国内国际两个大局，贯彻党的基本理论、基本路线、基本方略，统揽伟大斗争、伟大工程、伟大事业、伟大梦想，坚持稳中求进工作总基调，出台一系列重大方针政策，推出一系列重大举措，推进一系列重大工作，战胜一系列重大风险挑战，解决了许多长期想解决而没有解决的难题，办成了许多过去想办而没有办成的大事，推动党和国家事业取得历史性成就、发生历史性变革。全会强调，党的十八大以来，在坚持党的全面领导上，党中央权威和集中统一领导得到有力保证，党的领导制度体系不断完善，党的领导方式更加科学，全党思想上更加统一、政治上更加团结、行动上更加一致，党的政治领导力、思想引领力、群众组织力、社会号召力显著增强。在全面从严治党上，党的自我净化、自我完善、自我革新、自我提高能力显著增强，管党治党宽松软状况得到根本扭转，反腐败斗争取得压倒性胜利并全面巩固，党在革命性锻造中更加坚强。在经济建设上，我国经济发展平衡性、协调性、可持续性明显增强，国家经济实力、科技实力、综合国力跃上新台阶，我国经济迈上更高质量、更有效率、更加公平、更可持续、更为安全的发展之路。在全面深化改革开放上，党不断推动全面深化改革向广度和深度进军，中国特色社会主义制度更加成熟更加定型，国家治理体系和治理能力现代化水平不断提高，党和国家事业焕发出新的生机活力。在政治建设上，积极发展全过程人民民主，我国社会主义民主政治制度化、规范化、程序化全面推进，中国特色社会主义政治制度优越性得到更好发挥，生动活泼、安定团结的政治局面得到巩固和发展。在全面依法治国上，中国特色社会主义法治体系不断健全，法治中国建设迈出坚实步伐，党运用法治方式领导和治理国家的能力显著增强。在文化建设上，我国意识形态领域形势发生全局性、根本性转变，全党全国各族人民文化自信明显增强，全社会凝聚力和向心力极大提升，为新时代开创党和国家事业新局面提供了坚强思想保证和强大精神力量。在社会建设上，人民生活全方位改善，社会治理社会化、法治化、智能化、专业化水平大幅度提升，发展了人民安居乐业、社会安定有序的良好局面，续写了社会长期稳定奇迹。在生态文明建设上，党中央以前所未有的力度抓生态文明建设，美丽中国建设迈出重大步伐，我国生态环境保护发生历史性、转折性、全局性变化。在国防和军队建设上，人民军队实现整体性革命性重塑、重整行装再出发，国防实力和经济实力同步提升，人民军队坚决履行新时代使命任务，以顽强斗争精神和实际行动捍卫了国家主权、安全、发展利益。在维护国家安全上，国家安全得到全面加强，经受住了来自政治、经济、意识形态、自然界等方面的风险挑战考验，为党和国家兴旺发达、长治久安提供了有力保证。在坚持“一国两制”和推进祖国统一上，党中央采取一系列标本兼治的举措，坚定落实“爱国者治港”、“爱国者治澳”，推动香港局势实现由乱到治的重大转折，为推进依法治港治澳、促进“一国两制”实践行稳致远打下了坚实基础；坚持一个中国原则和“九二共识”，坚决反对“台独”分裂行径，坚决反对外部势力干涉，牢牢把握两岸关系主导权和主动权。在外交工作上，中国特色大国外交全面推进，构建人类命运共同体成为引领时代潮流和人类前进方向的鲜明旗帜，我国外交在世界大变局中开创新局、在世界乱局中化危为机，我国国际影响力、感召力、塑造力显著提升。中国共产党和中国人民以英勇顽强的奋斗向世界庄严宣告，中华民族迎来了从站起来、富起来到强起来的伟大飞跃。全会指出了中国共产党百年奋斗的历史意义：党的百年奋斗从根本上改变了中国人民的前途命运，中国人民彻底摆脱了被欺负、被压迫、被奴役的命运，成为国家、社会和自己命运的主人，中国人民对美好生活的向往不断变为现实；党的百年奋斗开辟了实现中华民族伟大复兴的正确道路，中国仅用几十年时间就走完发达国家几百年走过的工业化历程，创造了经济快速发展和社会长期稳定两大奇迹；党的百年奋斗展示了马克思主义的强大生命力，马克思主义的科学性和真理性在中国得到充分检验，马克思主义的人民性和实践性在中国得到充分贯彻，马克思主义的开放性和时代性在中国得到充分彰显；党的百年奋斗深刻影响了世界历史进程，党领导人民成功走出中国式现代化道路，创造了人类文明新形态，拓展了发展中国家走向现代化的途径；党的百年奋斗锻造了走在时代前列的中国共产党，形成了以伟大建党精神为源头的精神谱系，保持了党的先进性和纯洁性，党的执政能力和领导水平不断提高，中国共产党无愧为伟大光荣正确的党。全会提出，一百年来，党领导人民进行伟大奋斗，积累了宝贵的历史经验，这就是：坚持党的领导，坚持人民至上，坚持理论创新，坚持独立自主，坚持中国道路，坚持胸怀天下，坚持开拓创新，坚持敢于斗争，坚持统一战线，坚持自我革命。以上十个方面，是经过长期实践积累的宝贵经验，是党和人民共同创造的精神财富，必须倍加珍惜、长期坚持，并在新时代实践中不断丰富和发展。全会提出，不忘初心，方得始终。中国共产党立志于中华民族千秋伟业，百年恰是风华正茂。过去一百年，党向人民、向历史交出了一份优异的答卷。现在，党团结带领中国人民又踏上了实现第二个百年奋斗目标新的赶考之路。全党要牢记中国共产党是什么、要干什么这个根本问题，把握历史发展大势，坚定理想信念，牢记初心使命，始终谦虚谨慎、不骄不躁、艰苦奋斗，不为任何风险所惧，不为任何干扰所惑，决不在根本性问题上出现颠覆性错误，以咬定青山不放松的执着奋力实现既定目标，以行百里者半九十的清醒不懈推进中华民族伟大复兴。全会强调，全党必须坚持马克思列宁主义、毛泽东思想、邓小平理论、“三个代表”重要思想、科学发展观，全面贯彻习近平新时代中国特色社会主义思想，用马克思主义的立场、观点、方法观察时代、把握时代、引领时代，不断深化对共产党执政规律、社会主义建设规律、人类社会发展规律的认识。必须坚持党的基本理论、基本路线、基本方略，增强“四个意识”，坚定“四个自信”，做到“两个维护”，坚持系统观念，统筹推进“五位一体”总体布局，协调推进“四个全面”战略布局，立足新发展阶段、贯彻新发展理念、构建新发展格局、推动高质量发展，全面深化改革开放，促进共同富裕，推进科技自立自强，发展全过程人民民主，保证人民当家作主，坚持全面依法治国，坚持社会主义核心价值体系，坚持在发展中保障和改善民生，坚持人与自然和谐共生，统筹发展和安全，加快国防和军队现代化，协同推进人民富裕、国家强盛、中国美丽。全会强调，全党必须永远保持同人民群众的血肉联系，践行以人民为中心的发展思想，不断实现好、维护好、发展好最广大人民根本利益，团结带领全国各族人民不断为美好生活而奋斗。全党必须铭记生于忧患、死于安乐，常怀远虑、居安思危，继续推进新时代党的建设新的伟大工程，坚持全面从严治党，坚定不移推进党风廉政建设和反腐败斗争，做到难不住、压不垮，推动中国特色社会主义事业航船劈波斩浪、一往无前。全会决定，中国共产党第二十次全国代表大会于2022年下半年在北京召开。全会认为，党的二十大是我们党进入全面建设社会主义现代化国家、向第二个百年奋斗目标进军新征程的重要时刻召开的一次十分重要的代表大会，是党和国家政治生活中的一件大事。全党要团结带领全国各族人民攻坚克难、开拓奋进，为全面建设社会主义现代化国家、夺取新时代中国特色社会主义伟大胜利、实现中华民族伟大复兴的中国梦作出新的更大贡献，以优异成绩迎接党的二十大召开。党中央号召，全党全军全国各族人民要更加紧密地团结在以习近平同志为核心的党中央周围，全面贯彻习近平新时代中国特色社会主义思想，大力弘扬伟大建党精神，勿忘昨天的苦难辉煌，无愧今天的使命担当，不负明天的伟大梦想，以史为鉴、开创未来，埋头苦干、勇毅前行，为实现第二个百年奋斗目标、实现中华民族伟大复兴的中国梦而不懈奋斗。我们坚信，在过去一百年赢得了伟大胜利和荣光的中国共产党和中国人民，必将在新时代新征程上赢得更加伟大的胜利和荣光"))
