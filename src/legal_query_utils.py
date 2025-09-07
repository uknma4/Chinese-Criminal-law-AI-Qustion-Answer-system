import requests
import time
import numpy as np  # 新增numpy用于处理向量
import requests
import json
import os
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch, helpers
import logging
from typing import List, Dict, Any, Optional, Tuple
from huggingface_hub import snapshot_download
import logging
import threading
from text2vec import SentenceModel

# 配置日志记录
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# 初始化模型（缓存提升性能）
MODEL_CACHE = None


def load_model(model_name: str = "/home/eddie/models/textmodel"):
    """加载并缓存嵌入模型"""
    global MODEL_CACHE
    if MODEL_CACHE is None:
        logger.info(f"Loading sentence transformer model: {model_name}")
        MODEL_CACHE = SentenceModel(model_name)
    return MODEL_CACHE


# Elasticsearch 客户端配置
es = Elasticsearch(
    hosts=["http://localhost:9200"],
    retry_on_timeout=True,
    max_retries=3,
    request_timeout=30
)


def generate_query_vector(text: str) -> List[float]:
    """生成查询向量（带异常处理）"""
    try:
        model = load_model()
        return model.encode(text, normalize_embeddings=True).tolist()
    except Exception as e:
        logger.error(f"向量生成失败: {str(e)}")
        return []


def build_es_query(
        query: str,
        query_vector: List[float],
        top_k: int = 50,
        min_score: float = 0.5
) -> Dict:
    """
    构建 Elasticsearch 查询结构
    """
    field_weights = {
        "chapter_title": 1.2,
        "sections_title": 1.5,
        "subsections_title": 1.3,
        "article_title": 1.7,
        "article_content": 1.0
    }

    vector_config = {
        "section_title_vector": 0.6,
        "subsection_title_vector": 0.5,
        "article_title_vector": 0.7,
        "article_content_vector": 0.8
    }

    search_query = {
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": query,
                            "fields": [f"{field}^{weight}" for field, weight in field_weights.items()],
                            "type": "best_fields",
                            "tie_breaker": 0.3
                        }
                    },
                    {
                        "knn": {
                            "field": "article_content_vector",
                            "query_vector": query_vector,
                            "num_candidates": 100,
                            "boost": vector_config["article_content_vector"],
                            "similarity": 30
                        }
                    }
                    ,
                    # 向量搜索：章节标题向量（新增）
                    {
                        "knn": {
                            "field": "section_title_vector",
                            "query_vector": query_vector,  # 需单独生成标题向量
                            "num_candidates": 100,
                            "boost": vector_config["section_title_vector"]
                        }
                    }
                ],
                "minimum_should_match": 1,
                "boost": 1.0
            }
        },
        "_source": ["chapter_title", "sections_title", "subsections_title", "article_title", "article_content"],
        "min_score": min_score,
        "explain": True
    }

    return search_query


def query_es_content(
        index_name: str,
        content_question: str,
        top_k: int = 50,
        min_score: float = 0.5
) -> List[Dict[str, Any]]:
    """
    执行 Elasticsearch 查询，返回结果
    """
    query_vector = generate_query_vector(content_question)
    if not query_vector:
        return []

    search_query = build_es_query(
        query=content_question,
        query_vector=query_vector,
        top_k=top_k,
        min_score=min_score
    )

    try:
        logger.info(f"Executing search on index: {index_name}")
        response = es.search(
            index=index_name,
            body=search_query,
            request_timeout=45
        )
        logger.debug(f"Search completed in {response['took']}ms")
    except Exception as e:
        logger.error(f"搜索失败: {str(e)}")
        return []

    return process_search_results(response, min_score)


def process_search_results(
        response: Dict,
        min_score: float
) -> List[Dict[str, Any]]:
    """处理和分析搜索结果"""
    hits = response.get("hits", {}).get("hits", [])

    scores = [hit["_score"] for hit in hits]
    logger.info(f"""
    评分分析：
    - 平均分: {np.mean(scores):.2f}
    - 最高分: {np.max(scores):.2f}
    - 最低分: {np.min(scores):.2f}
    - 有效结果: {len([s for s in scores if s >= min_score])}/{len(scores)}
    """)

    results = []
    for hit in hits:
        if hit["_score"] < min_score:
            continue

        source = hit.get("_source", {})
        results.append({
            "id": hit["_id"],
            "score": hit["_score"],
            "chapter_title": source.get("chapter_title", ""),
            "sections_title": source.get("sections_title", ""),
            "subsections_title": source.get("subsections_title", ""),
            "article_title": source.get("article_title", ""),
            "article_content": source.get("article_content", "")[:3000],
            "explanation": hit.get("_explanation")
        })

    return results


def query_ollama(model, prompt, context) -> List[Dict[str, Any]]:
    """简化版 Ollama 查询函数"""
    url = "http://localhost:11434/api/chat"
    headers = {"Content-Type": "application/json"}

    data = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": f"你是一个法律专家，请基于以下和问题相关的内容回答：\n{context}"
            },
            {"role": "user", "content": prompt}
        ],
        "stream": False
    }

    response = requests.post(url, headers=headers, json=data)
    return response.json()
    # 2. 构建上下文


