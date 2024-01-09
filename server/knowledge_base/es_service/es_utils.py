import time
from datetime import datetime
from typing import Dict

from elasticsearch import Elasticsearch
from langchain.document_loaders import TextLoader
from langchain.schema.embeddings import Embeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from configs import kbs_config, logger
from server.query_process.base import query_time_extract
import jionlp as jio


def load_file(filepath, chunk_size, chunk_overlap):
    loader = TextLoader(filepath, encoding='utf-8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs


def _default_knn_setting(dim: int) -> Dict:
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 1,
            "analysis": {
                "analyzer": {
                    "ik_analyzer": {
                        "type": "custom",
                        "tokenizer": "ik_max_word"
                    }
                }
            }
        },
        "mappings": {
            # "dynamic": "strict",
            "properties": {
                "text": {"type": "text"
                    , "analyzer": "ik_analyzer"},
                "vector": {
                    "type": "dense_vector",
                    "dims": dim,
                    "index": True,
                    "similarity": "cosine",
                },
                "time": {
                    "type": "long"
                },
            }
        }
    }


def generate_search_query(text: str,
                          embedding: Embeddings,
                          size: int,
                          time_filter: bool = False) -> Dict:
    # time_out_text, timeInfo = query_time_extract(text)
    timeInfo = jio.ner.extract_time(text)
    vec = embedding.embed_query(text)
    if not time_filter or not timeInfo:
        query = {
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.queryVector, 'vector') + 1.0",
                        "params": {
                            "queryVector": vec
                        }
                    }
                }
            },
            "size": size
        }
        return query
    time_range = []
    for time in timeInfo:
        if time["type"] == "time_span" or time["type"] == "time_point":
            time_range.append({
                "range": {
                    "time": {
                        "gte": datetime.strptime(time["detail"]["time"][0], "%Y-%m-%d %H:%M:%S").timestamp(),
                        "lte": datetime.strptime(time["detail"]["time"][1], "%Y-%m-%d %H:%M:%S").timestamp()
                    }
                }})
    query = {
        "query": {
            "bool": {
                "must": {
                    "script_score": {
                        "query": {
                            "match_all": {}
                        },
                        "script": {
                            "source": "cosineSimilarity(params.queryVector, 'vector') + 1.0",
                            "params": {
                                "queryVector": vec
                            }
                        }
                    }
                },
                "filter": {
                    "bool": {
                        "should": time_range
                    }
                }
            }
        },
        "size": size
    }
    return query


def generate_knn_query(text: str,
                       embedding: Embeddings,
                       size: int,
                       time_filter: bool = False) -> Dict:
    timeInfo = jio.ner.extract_time(text)
    vec = embedding.embed_query(text)
    if not time_filter or not timeInfo:
        query = {
            "knn": {
                "field": "vector",
                "query_vector": vec,
                "k": 10,
                "num_candidates": 100
            },
            "size": size
        }
        return query
    time_range = []
    for time in timeInfo:
        if time["type"] == "time_span" or time["type"] == "time_point":
            time_range.append({
                "range": {
                    "time": {
                        "gte": datetime.strptime(time["detail"]["time"][0], "%Y-%m-%d %H:%M:%S").timestamp(),
                        "lte": datetime.strptime(time["detail"]["time"][1], "%Y-%m-%d %H:%M:%S").timestamp()
                    }
                }})
    query_time_range = {
        "knn": {
            "field": "vector",
            "query_vector": vec,
            "k": 10,
            "num_candidates": size * 5,
            "filter": {
                "bool": {
                    "should": time_range
                }
            }
        },
        "size": size
    }
    return query_time_range


def generate_hybrid_query(text: str,
                          embedding: Embeddings,
                          size: int = 10,
                          time_filter: bool = False) -> Dict:
    timeInfo = jio.ner.extract_time(text)
    vec = embedding.embed_query(text)
    if not time_filter or not timeInfo:
        query = {
            "query": {
                "match": {
                    "text": {
                        "query": text,
                    }
                }
            },
            "knn": {
                "field": "vector",
                "query_vector": vec,
                "k": 10,
                "num_candidates": size * 5,
            },
            "size": size,
            # "rank": {"rrf": {}}
        }
        return query
    logger.info(f"启用时间过滤")
    logger.info(f"query时间信息是{timeInfo}")
    time_range = []
    for time in timeInfo:
        if time["type"] == "time_span" or time["type"] == "time_point":
            time_range.append({
                "range": {
                    "time": {
                        "gte": datetime.strptime(time["detail"]["time"][0], "%Y-%m-%d %H:%M:%S").timestamp(),
                        "lte": datetime.strptime(time["detail"]["time"][1], "%Y-%m-%d %H:%M:%S").timestamp()
                    }
                }})
    query = {
        "query": {
            "bool": {
                "must": {
                    "match": {
                        "text": {
                            "query": text
                        }
                    }
                },
                "filter": {
                    "bool": {
                        "should": time_range
                    }
                }
            }
        },
        "knn": {
            "field": "vector",
            "query_vector": vec,
            "k": 10,
            "num_candidates": size * 5,
            "filter": {
                "bool": {
                    "should": time_range
                }
            }
        },
        "size": size,
        # "rank": {"rrf": {}}
    }
    return query


def generate_keywords_query(text: str,
                            size: int,
                            time_filter: bool = False) -> Dict:
    timeInfo = jio.ner.extract_time(text)
    if not time_filter or not timeInfo:
        query = {
            "query": {
                "match": {
                    "text": {
                        "query": text,
                    }
                }
            },
            "size": size,
        }
        return query
    logger.info(f"启用时间过滤")
    logger.info(f"query时间信息是{timeInfo}")
    time_range = []
    for time in timeInfo:
        if time["type"] == "time_span" or time["type"] == "time_point":
            time_range.append({
                "range": {
                    "time": {
                        "gte": datetime.strptime(time["detail"]["time"][0], "%Y-%m-%d %H:%M:%S").timestamp(),
                        "lte": datetime.strptime(time["detail"]["time"][1], "%Y-%m-%d %H:%M:%S").timestamp()
                    }
                }})
    query = {
        "query": {
            "bool": {
                "must": {
                    "match": {
                        "text": {
                            "query": text
                        }
                    }
                },
                "filter": {
                    "bool": {
                        "should": time_range
                    }
                }
            }
        },
        "size": size,
    }
    return query


es_params = kbs_config["es"]
host = 'https://{}:{}'.format(es_params['host'], es_params['port'])
es_client = Elasticsearch([host],
                          basic_auth=(es_params['username'], es_params['passwd']),
                          ca_certs=es_params['ca_certs_path'], )
