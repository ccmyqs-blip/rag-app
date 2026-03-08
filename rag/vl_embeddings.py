import os
from typing import List

from langchain_core.embeddings import Embeddings

from utils.logger_handler import logger


def _extract_embedding_from_dashscope_response(resp) -> list[float]:
    """
    尽量兼容不同 dashscope SDK 版本的返回结构。
    期望返回单条 embedding 向量。
    """
    out = None
    try:
        out = resp.output
    except Exception:
        out = None

    if isinstance(out, dict):
        # 常见结构 1: {"embeddings":[{"embedding":[...]}]}
        emb_list = out.get("embeddings")
        if isinstance(emb_list, list) and emb_list:
            emb = emb_list[0].get("embedding")
            if isinstance(emb, list):
                return emb

        # 常见结构 2: {"output":{"embeddings":[...]}}
        inner = out.get("output")
        if isinstance(inner, dict):
            emb_list = inner.get("embeddings")
            if isinstance(emb_list, list) and emb_list:
                emb = emb_list[0].get("embedding")
                if isinstance(emb, list):
                    return emb

        # 常见结构 3: {"data":[{"embedding":[...]}]}
        data = out.get("data")
        if isinstance(data, list) and data:
            emb = data[0].get("embedding")
            if isinstance(emb, list):
                return emb

    # 兜底：尝试属性访问
    try:
        out2 = resp.output
        emb_list = getattr(out2, "embeddings", None)
        if isinstance(emb_list, list) and emb_list:
            emb = getattr(emb_list[0], "embedding", None)
            if isinstance(emb, list):
                return emb
    except Exception:
        pass

    raise ValueError("无法从 dashscope 返回中解析 embedding")


class Qwen3VLEmbeddings(Embeddings):
    """
    使用 DashScope 多模态 embedding 模型，将文本嵌入到与图片同一向量空间。
    用于：文本->文本、文本->图片 的统一检索。
    """

    def __init__(self, model_name: str = "qwen3-vl-embedding"):
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_text(t, text_type="document") for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_text(text, text_type="query")

    def _embed_text(self, text: str, text_type: str) -> List[float]:
        api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("未检测到环境变量 DASHSCOPE_API_KEY，无法调用 qwen3-vl-embedding")

        try:
            from dashscope.embeddings.multimodal_embedding import (  # type: ignore
                MultiModalEmbedding,
                MultiModalEmbeddingItemText,
            )

            resp = MultiModalEmbedding.call(
                model=self.model_name,
                input=[MultiModalEmbeddingItemText(text=text, factor=1.0)],
                api_key=api_key,
                auto_truncation=True,
                text_type=text_type,
            )
            return _extract_embedding_from_dashscope_response(resp)
        except Exception as e:
            logger.error(f"qwen3-vl-embedding 调用失败：{type(e).__name__}: {e}")
            raise

