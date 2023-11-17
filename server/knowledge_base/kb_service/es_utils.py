from typing import Dict

from elasticsearch import Elasticsearch
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.elasticsearch import ElasticsearchStore
from configs import kbs_config


def load_file(filepath, chunk_size, chunk_overlap):
    loader = TextLoader(filepath, encoding='utf-8')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs


def _default_knn_setting(dim: int) -> Dict:
    """Generates a default index mapping for kNN search."""
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
            }
        }
    }


def generate_search_query(vec, size) -> Dict:
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


def generate_knn_query(vec, size) -> Dict:
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


def generate_hybrid_query(text, vec, size) -> Dict:
    query = {
        "query": {
            "match": {
                "text": {
                    "query": text,
                    # "boost": 1 - knn_boost
                }
            }
        },
        "knn": {
            "field": "vector",
            "query_vector": vec,
            "k": 10,
            "num_candidates": size * 5,
            # "boost": knn_boost
        },
        "size": size,
        "rank": {"rrf": {}}
    }
    return query


def generate_keywords_query(text, size) -> Dict:
    query = {
        "query": {
            "match": {
                "text": {
                    "query": text,
                }
            }
        },
        "size": size,
        # "rank": {"rrf": {}}
    }
    return query



es_params = kbs_config["es"]
host = 'https://{}:{}'.format(es_params['host'], es_params['port'])
es_client = Elasticsearch([host],
                          basic_auth=(es_params['username'], es_params['passwd']),
                          ca_certs=es_params['ca_certs_path'], )
