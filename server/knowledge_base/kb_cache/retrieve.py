

def retrieve(embeddings, query, top_k, score_threshold, search_method, use_rerank):
    docs = []
    top_k_1 = top_k * 5 if use_rerank else top_k

    # 召回阶段
    if search_method == "hybrid":
        docsCos = self.do_search(query=query, top_k=top_k_1, score_threshold=score_threshold,
                                 embeddings=embeddings,
                                 method="cos")
        docsBM25 = self.do_search(query=query, top_k=top_k_1, score_threshold=score_threshold,
                                  embeddings=embeddings,
                                  method="keywords")
        docs = []
        exist_doc = set()
        ## 去重
        for doc in docsCos:
            doc_id = doc[0].metadata["source"] + "_" + doc[0].metadata["chunk"]
            if doc_id not in exist_doc:
                docs.append(doc)
                exist_doc.add(doc_id)
        for doc in docsBM25:
            doc_id = doc[0].metadata["source"] + "_" + doc[0].metadata["chunk"]
            if doc_id not in exist_doc:
                docs.append(doc)
                exist_doc.add(doc_id)

    elif search_method == "cos":
        docs = self.do_search(query=query, top_k=top_k_1, score_threshold=score_threshold,
                              embeddings=embeddings,
                              method="cos")
    elif search_method == "keywords":
        docs = self.do_search(query=query, top_k=top_k_1, score_threshold=score_threshold,
                              embeddings=embeddings,
                              method="keywords")
    logger.info("召回阶段完成，一共召回了{}个结果".format(len(docs)))