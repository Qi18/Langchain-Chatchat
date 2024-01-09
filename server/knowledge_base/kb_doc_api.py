import asyncio
import os
import urllib
from io import BytesIO

from fastapi import File, Form, Body, Query, UploadFile
from configs import (DEFAULT_VS_TYPE, EMBEDDING_MODEL,
                     VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD,
                     CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE,
                     logger, log_verbose, )
from server.utils import BaseResponse, ListResponse, run_in_thread_pool
from server.knowledge_base.utils import (validate_kb_name, list_files_from_folder, get_file_path,
                                         files2docs_in_thread, KnowledgeFile, get_kb_path)
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import Json, BaseModel
import json
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.db.repository.knowledge_file_repository import get_file_detail, add_files_to_db
from typing import List, Dict, Set
from langchain.docstore.document import Document
from server.chat.chat import chatRefine


class DocumentWithScore(Document):
    score: float = None


def search_docs(query: str = Body(..., description="用户输入", examples=["你好"]),
                knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                score_threshold: float = Body(SCORE_THRESHOLD,
                                              description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                                              ge=0, le=1),
                search_method: str = Body("hybrid", description="搜索方法，可选值为knn, hybrid, cos"),
                ) -> List[DocumentWithScore]:
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return []
    docs = kb.search_docs(query, top_k, score_threshold, search_method)
    data = [DocumentWithScore(**x[0].dict(), score=x[1]) for x in docs]
    return data


def search_docs_multiQ(querys: List = Body(..., description="用户输入", examples=["你好"]),
                       knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                       top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                       score_threshold: float = Body(SCORE_THRESHOLD,
                                                     description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                                                     ge=0, le=1),
                       search_method: str = Body("hybrid", description="搜索方法，可选值为knn, hybrid, cos"),
                       ) -> List[DocumentWithScore]:
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None or len(querys) == 0:
        return []
    docs = kb.search_docs_multiQ(querys, top_k, score_threshold, search_method)
    data = [DocumentWithScore(**x[0].dict(), score=x[1]) for x in docs]
    return data


def list_files(
        knowledge_base_name: str
) -> ListResponse:
    if not validate_kb_name(knowledge_base_name):
        return ListResponse(code=403, msg="Don't attack me", data=[])

    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return ListResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}", data=[])
    else:
        all_doc_names = kb.list_files()
        return ListResponse(data=all_doc_names)


def _save_files_in_thread(files: List[UploadFile],
                          knowledge_base_name: str,
                          override: bool):
    '''
    通过多线程将上传的文件保存到对应知识库目录内。
    生成器返回保存结果：{"code":200, "msg": "xxx", "data": {"knowledge_base_name":"xxx", "file_name": "xxx"}}
    '''

    def save_file(file: UploadFile, knowledge_base_name: str, override: bool) -> dict:
        '''
        保存单个文件。
        '''
        try:
            filename = file.filename
            file_path = get_file_path(knowledge_base_name=knowledge_base_name, doc_name=filename)
            data = {"knowledge_base_name": knowledge_base_name, "file_name": filename}

            file_content = file.file.read()  # 读取上传文件的内容
            if (os.path.isfile(file_path)
                    and not override
                    and os.path.getsize(file_path) == len(file_content)
            ):
                # TODO: filesize 不同后的处理
                file_status = f"文件 {filename} 已存在。"
                logger.warn(file_status)
                return dict(code=404, msg=file_status, data=data)

            with open(file_path, "wb") as f:
                f.write(file_content)
            return dict(code=200, msg=f"成功上传文件 {filename}", data=data)
        except Exception as e:
            msg = f"{filename} 文件上传失败，报错信息为: {e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            return dict(code=500, msg=msg, data=data)

    params = [{"file": file, "knowledge_base_name": knowledge_base_name, "override": override} for file in files]
    for result in run_in_thread_pool(save_file, params=params):
        yield result


# 似乎没有单独增加一个文件上传API接口的必要
# def upload_files(files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
#                 knowledge_base_name: str = Form(..., description="知识库名称", examples=["samples"]),
#                 override: bool = Form(False, description="覆盖已有文件")):
#     '''
#     API接口：上传文件。流式返回保存结果：{"code":200, "msg": "xxx", "data": {"knowledge_base_name":"xxx", "file_name": "xxx"}}
#     '''
#     def generate(files, knowledge_base_name, override):
#         for result in _save_files_in_thread(files, knowledge_base_name=knowledge_base_name, override=override):
#             yield json.dumps(result, ensure_ascii=False)

#     return StreamingResponse(generate(files, knowledge_base_name=knowledge_base_name, override=override), media_type="text/event-stream")


# TODO: 等langchain.document_loaders支持内存文件的时候再开通
# def files2docs(files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
#                 knowledge_base_name: str = Form(..., description="知识库名称", examples=["samples"]),
#                 override: bool = Form(False, description="覆盖已有文件"),
#                 save: bool = Form(True, description="是否将文件保存到知识库目录")):
#     def save_files(files, knowledge_base_name, override):
#         for result in _save_files_in_thread(files, knowledge_base_name=knowledge_base_name, override=override):
#             yield json.dumps(result, ensure_ascii=False)

#     def files_to_docs(files):
#         for result in files2docs_in_thread(files):
#             yield json.dumps(result, ensure_ascii=False)


def upload_files(files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
                 knowledge_base_name: str = Form(..., description="知识库名称", examples=["samples"]),
                 override: bool = Form(False, description="覆盖已有文件"),
                 to_vector_store: bool = Form(True, description="上传文件后是否进行向量化"),
                 chunk_size: int = Form(CHUNK_SIZE, description="知识库中单段文本最大长度"),
                 chunk_overlap: int = Form(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
                 zh_title_enhance: bool = Form(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
                 not_refresh_vs_cache: bool = Form(False, description="暂不保存向量库（用于FAISS）"),
                 enhanceOperation: List[str] = Form(None,
                                                    description="额外的增强操作，summary表示在上传时添加一个总结文档"),
                 ) -> BaseResponse:
    '''
    API接口：上传文件，并/或向量化
    '''
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    failed_files = {}
    file_names = []

    # 先将上传的文件保存到磁盘
    for result in _save_files_in_thread(files, knowledge_base_name=knowledge_base_name, override=override):
        filename = result["data"]["file_name"]
        if result["code"] != 200:
            failed_files[filename] = result["msg"]
            continue
        if filename not in file_names:
            file_names.append(filename)

    # 对保存的文件进行向量化
    if to_vector_store:
        result = update_files(
            knowledge_base_name=knowledge_base_name,
            file_names=file_names,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            zh_title_enhance=zh_title_enhance,
            not_refresh_vs_cache=True,
            enhanceOperation=[] if enhanceOperation is None else enhanceOperation
        )
        failed_files.update(result.data["failed_files"])
        if not not_refresh_vs_cache:
            kb.save_vector_store()

    return BaseResponse(code=200, msg="文件上传与向量化完成", data={"failed_files": failed_files})


def delete_files(knowledge_base_name: str = Body(..., examples=["samples"]),
                 file_names: List[str] = Body(..., examples=[["file_name.md", "test.txt"]]),
                 delete_content: bool = Body(False),
                 not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
                 ) -> BaseResponse:
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    failed_files = {}
    for file_name in file_names:
        if not kb.exist_doc(file_name):
            failed_files[file_name] = f"未找到文件 {file_name}"

        try:
            kb_file = KnowledgeFile(filename=file_name,
                                    knowledge_base_name=knowledge_base_name)
            kb.delete_doc(kb_file, delete_content, not_refresh_vs_cache=True)
        except Exception as e:
            msg = f"{file_name} 文件删除失败，错误信息：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            failed_files[file_name] = msg

        # 对对应的summary文件进行删除
        summary_file = file_name + "_summary.txt"
        if not kb.exist_doc(file_name):
            continue
        try:
            kb_file = KnowledgeFile(filename=summary_file,
                                    knowledge_base_name=knowledge_base_name)
            kb.delete_doc(kb_file, delete_content, not_refresh_vs_cache=True)
        except Exception as e:
            msg = f"{summary_file} 文件删除失败，错误信息：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)

    if not not_refresh_vs_cache:
        kb.save_vector_store()

    return BaseResponse(code=200, msg=f"文件删除完成", data={"failed_files": failed_files})


def update_files(
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        file_names: List[str] = Body(..., description="文件名称，支持多文件", examples=[["file_name1", "text.txt"]]),
        file_metadata: Json = Body(None, description="文件的metadata，支持多文件", examples=[{"test.txt": {"key1": "value1"}}]),
        chunk_size: int = Body(CHUNK_SIZE, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Body(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
        zh_title_enhance: bool = Body(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
        not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
        enhanceOperation: List[str] = Form(None,
                                           description="额外的增强操作，summary表示在上传时添加一个总结文档", examples=["summary"]),
) -> BaseResponse:
    '''
    更新知识库文档
    '''
    # print(file_metadata)
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    failed_files = {}
    kb_files = []

    # 生成需要加载docs的文件列表
    for file_name in file_names:
        try:
            if file_metadata and file_name in file_metadata.keys():
                kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name
                                        , chunk_overlap=chunk_overlap, chunk_size=chunk_size,
                                        zh_title_enhance=zh_title_enhance, metadata=file_metadata[file_name])
            else:
                kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name
                                        , chunk_overlap=chunk_overlap, chunk_size=chunk_size,
                                        zh_title_enhance=zh_title_enhance)
            # 判断文件的名字长度是否太长
            os.path.getmtime(kb_file.filepath)
            kb_files.append(kb_file)
        except Exception as e:
            msg = f"加载文档 {file_name} 时出错：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            failed_files[file_name] = msg

    # 向向量库和metadata数据库添加文件信息
    kb.update_docs(kb_files)

    # TODO 异步总结所有的file，并上传到向量库和meta数据库
    if enhanceOperation is not None and "summary" in enhanceOperation:
        # asyncio.run(summary_tasks(kb_files=kb_files, knowledge_base_name=knowledge_base_name))
        # summary_files = {}
        for kb_file_group in [kb_files[index: index + 20] for index in range(0, len(kb_files), 20)]:
            summary_files = {}
            for kb_file in kb_file_group:
                if kb_file.file2full_text() is None or kb_file.file2full_text() == "" or len(kb_file.file2text()) < 2:
                    continue
                query = f"文章标题是{kb_file.filename}\n文章内容是{kb_file.file2full_text().page_content}"
                summary = chatRefine(query)

                name = os.path.splitext(kb_file.filename)[0]
                summary_files[name + "_summary"] = summary

            # summary文档一定会改写之前的文档
            upload_custom_files(files=summary_files, knowledge_base_name=knowledge_base_name, override=True,
                                enhanceOperation=[])

    if not not_refresh_vs_cache:
        kb.save_vector_store()

    return BaseResponse(code=200, msg=f"更新文档完成", data={"failed_files": failed_files})


def upload_custom_files(
        files: Json = Form(..., description="自定义的docs，需要转为json字符串",
                           examples=[{"test": {"content":"这个一个自定义的doc", "metadata": {"key1": "value1"}}}]),
        knowledge_base_name: str = Form(..., description="知识库名称", examples=["samples"]),
        override: bool = Form(True, description="覆盖已有文件"),
        to_vector_store: bool = Form(True, description="上传文件后是否进行向量化"),
        chunk_size: int = Form(CHUNK_SIZE, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Form(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
        zh_title_enhance: bool = Form(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
        not_refresh_vs_cache: bool = Form(False, description="暂不保存向量库（用于FAISS）"),
        enhanceOperation: List[str] = Form(None,
                                           description="额外的增强操作，summary表示在上传时添加一个总结文档"),
) -> BaseResponse:
    # 上传自定义的files会将files保存为txt文件
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    failed_files = {}
    final_file_names = []

    # 先将上传的文件保存到磁盘
    for result in save_custom_files(files=files, knowledge_base_name=knowledge_base_name, override=override):
        filename = result["data"]["file_name"]
        if result["code"] != 200:
            failed_files[filename] = result["msg"]
            continue
        final_file_names.append(str(filename) + ".txt")

    file_metadata = {}
    for filename in files.keys():
        if "metadata" in files[filename].keys() and files[filename]["metadata"]:
            file_metadata[filename + ".txt"] = files[filename]["metadata"]
   #对保存的文件进行向量化
    if to_vector_store:
        result = update_files(
            knowledge_base_name=knowledge_base_name,
            file_names=final_file_names,
            file_metadata=file_metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            zh_title_enhance=zh_title_enhance,
            not_refresh_vs_cache=True,
            enhanceOperation=enhanceOperation
        )
        failed_files.update(result.data["failed_files"])
        if not not_refresh_vs_cache:
            kb.save_vector_store()

    return BaseResponse(code=200, msg="文件上传与向量化完成", data={"failed_files": failed_files})


def download_doc(
        knowledge_base_name: str = Query(..., description="知识库名称", examples=["samples"]),
        file_name: str = Query(..., description="文件名称", examples=["test.txt"]),
        preview: bool = Query(False, description="是：浏览器内预览；否：下载"),
):
    '''
    下载知识库文档
    '''
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    if preview:
        content_disposition_type = "inline"
    else:
        content_disposition_type = None

    try:
        kb_file = KnowledgeFile(filename=file_name,
                                knowledge_base_name=knowledge_base_name)

        if os.path.exists(kb_file.filepath):
            return FileResponse(
                path=kb_file.filepath,
                filename=kb_file.filename,
                media_type="multipart/form-data",
                content_disposition_type=content_disposition_type,
            )
    except Exception as e:
        msg = f"{kb_file.filename} 读取文件失败，错误信息是：{e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=500, msg=f"{kb_file.filename} 读取文件失败")


def recreate_vector_store(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        allow_empty_kb: bool = Body(True),
        vs_type: str = Body(DEFAULT_VS_TYPE),
        embed_model: str = Body(EMBEDDING_MODEL),
        chunk_size: int = Body(CHUNK_SIZE, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Body(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
        zh_title_enhance: bool = Body(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
        not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
):
    '''
    recreate vector store from the content.
    this is usefull when user can copy files to content folder directly instead of upload through network.
    by default, get_service_by_name only return knowledge base in the info.db and having document files in it.
    set allow_empty_kb to True make it applied on empty knowledge base which it not in the info.db or having no documents.
    '''

    def output():
        kb = KBServiceFactory.get_service(knowledge_base_name, vs_type, embed_model)
        if not kb.exists() and not allow_empty_kb:
            yield {"code": 404, "msg": f"未找到知识库 ‘{knowledge_base_name}’"}
        else:
            kb.create_kb()
            kb.clear_vs()
            files = list_files_from_folder(knowledge_base_name)
            kb_files = [(file, knowledge_base_name) for file in files]
            i = 0
            db_data, db_files = [], []
            for status, result in files2docs_in_thread(kb_files,
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap,
                                                       zh_title_enhance=zh_title_enhance):
                if status:
                    kb_name, file_name, docs = result
                    kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=kb_name)
                    kb_file.splited_docs = docs
                    yield json.dumps({
                        "code": 200,
                        "msg": f"({i + 1} / {len(files)}): {file_name}",
                        "total": len(files),
                        "finished": i,
                        "doc": file_name,
                    }, ensure_ascii=False)
                    custom_docs, length_docs, doc_infos = kb.add_doc(kb_file, not_refresh_vs_cache=True)
                    if length_docs != 0:
                        db_data.append({"custom_docs": custom_docs, "length_docs": length_docs, "doc_infos": doc_infos})
                        db_files.append(kb_file)
                else:
                    kb_name, file_name, error = result
                    msg = f"添加文件‘{file_name}’到知识库‘{knowledge_base_name}’时出错：{error}。已跳过。"
                    logger.error(msg)
                    yield json.dumps({
                        "code": 500,
                        "msg": msg,
                    })
                i += 1
            ## 将meta信息统一写入数据库
            add_files_to_db(kb_files=db_files, kb_data=db_data)
            if not not_refresh_vs_cache:
                kb.save_vector_store()

    return StreamingResponse(output(), media_type="text/event-stream")


def save_custom_files(files: json, knowledge_base_name: str, override: bool = False) -> list:
    result = []
    if not os.path.isdir(get_kb_path(knowledge_base_name=knowledge_base_name)):
        os.mkdir(get_kb_path(knowledge_base_name=knowledge_base_name))
    for filename in files.keys():
        try:
            file_path = get_file_path(knowledge_base_name=knowledge_base_name, doc_name=filename + ".txt")
            data = {"knowledge_base_name": knowledge_base_name, "file_name": filename}

            file_content = files[filename]["content"]
            if not file_content:
                continue
            if (os.path.isfile(file_path)
                    and not override
            ):
                file_status = f"文件 {filename} 已存在。"
                logger.warn(file_status)
                result.append(dict(code=404, msg=file_status, data=data))
                continue

            with open(file_path, "w") as f:
                f.write(file_content)
            result.append(dict(code=200, msg=f"成功上传文件 {filename}", data=data))
        except Exception as e:
            msg = f"{filename} 文件上传失败，报错信息为: {e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            result.append(dict(code=500, msg=msg, data=data))
    return result


async def summary_tasks(kb_files: List[KnowledgeFile],
                        knowledge_base_name: str, ):
    tasks = []
    semaphore = asyncio.Semaphore(4)  # 设置最大并发数量为4
    fileGroups = [[kb_files[i] for i in range(start, end)] for start, end in
                  zip(range(0, len(kb_files), 4), range(4, len(kb_files) + 1, 4))]  # file按4的数量分组
    for fileGroup in fileGroups:
        task = asyncio.create_task(
            summary_task(semaphore=semaphore, kb_files=fileGroup, knowledge_base_name=knowledge_base_name))
        tasks.append(task)
    await asyncio.gather(*tasks)


async def summary_task(
        semaphore: asyncio.Semaphore,
        kb_files: List[KnowledgeFile],
        knowledge_base_name: str, ):
    async with semaphore:
        print("开始执行总结任务")
        if not validate_kb_name(knowledge_base_name):
            return BaseResponse(code=403, msg="Don't attack me")

        kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
        if kb is None:
            return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
        summary_files = {}
        for kb_file in kb_files:
            if kb_file.file2full_text() is None or kb_file.file2full_text() == "":
                continue
            query = f"文章标题是{kb_file.filename}\n文章内容是{kb_file.file2full_text()}"
            summary = chatRefine(query)

            name = os.path.splitext(kb_file.filename)[0]
            summary_files[name + "_summary.txt"] = summary

        # summary文档一定会改写之前的文档
        upload_custom_files(files=summary_files, knowledge_base_name=knowledge_base_name, override=True,
                            enhanceOperation=[])
        print("任务执行完毕")


if __name__ == "__main__":
    upload_custom_files({"test": "a"}, "习近平重要讲话数据库")
