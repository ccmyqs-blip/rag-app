import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma

from utils.config_handler import chroma_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from rag.vl_embeddings import Qwen3VLEmbeddings, _extract_embedding_from_dashscope_response


@dataclass
class ImageRef:
    """
    图像检索结果的轻量封装。

    - md5: 入库时用于查重的 md5 值
    - source: 原始文件名
    - path: 本地图片路径（用于后续读取 bytes）
    - score: 相似度/距离分数（越小越相似，具体含义由 Chroma 配置决定）
    - metadata: Chroma 中存储的原始 metadata
    """

    md5: str
    source: str
    path: str
    score: Optional[float]
    metadata: Dict[str, Any]


class ImageRetrievalService:
    """
    基于现有 qwen3-vl-embedding + Chroma 的图片相似检索服务。

    约定：
    - 底层使用 UploadIngestService 写入的 *_vl collection
    - 图片向量直接存储在底层 collection 中，metadata 至少包含：
      - md5: 文件 md5
      - source: 原始文件名
      - type: "image"
    - 原始图片文件保存在 data/uploads 下，命名为 {md5}_{filename}
    """

    def __init__(self, collection_name: Optional[str] = None) -> None:
        vl_collection = collection_name or f"{chroma_conf['collection_name']}_vl"
        self.collection_name = vl_collection
        self._embedding_model_name = "qwen3-vl-embedding"

        # 虽然我们在相似度检索时会手动传入向量，但这里仍需要提供 embedding_function
        # 以保持 Chroma 封装的一致性。
        self.vector_store = Chroma(
            collection_name=vl_collection,
            embedding_function=Qwen3VLEmbeddings(model_name=self._embedding_model_name),
            persist_directory=chroma_conf["persist_directory"],
        )

        # 与 UploadIngestService 中 _save_original 保持一致的上传目录
        self._upload_dir = get_abs_path(os.path.join("data", "uploads"))
        self._tmp_dir = get_abs_path(os.path.join("data", "tmp"))

    def _embed_image_bytes(self, image_bytes: bytes) -> List[float]:
        """
        使用 qwen3-vl-embedding 对上传的目标图片生成向量。

        这里通过临时文件 + file URI 的方式调用 dashscope，多数版本兼容性最好。
        """
        api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("未检测到环境变量 DASHSCOPE_API_KEY，无法对图片生成 embedding")

        from dashscope.embeddings.multimodal_embedding import (  # type: ignore
            MultiModalEmbedding,
            MultiModalEmbeddingItemImage,
        )

        os.makedirs(self._tmp_dir, exist_ok=True)
        tmp_path = os.path.join(self._tmp_dir, f"query_{uuid.uuid4().hex}.png")
        with open(tmp_path, "wb") as f:
            f.write(image_bytes)

        try:
            image_uri = Path(tmp_path).resolve().as_uri()
            resp = MultiModalEmbedding.call(
                model=self._embedding_model_name,
                input=[MultiModalEmbeddingItemImage(image=image_uri, factor=1.0)],
                api_key=api_key,
                auto_truncation=True,
            )
            return _extract_embedding_from_dashscope_response(resp)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                # 临时文件删除失败对功能无影响，记录日志即可
                logger.warning("删除临时图片失败：%s", tmp_path)

    def _resolve_image_path(self, md5_hex: str, source: str) -> str:
        """
        根据 UploadIngestService 的约定，推导图片在磁盘上的路径。
        """
        if not md5_hex or not source:
            return ""
        safe_name = os.path.basename(source)
        return os.path.join(self._upload_dir, f"{md5_hex}_{safe_name}")

    def retrieve_similar_images(
        self,
        target_image_bytes: bytes,
        top_k: int = 6,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ImageRef]:
        """
        对单张目标图片做相似检索，返回若干参考图片。

        - target_image_bytes: 待检索图片的二进制内容
        - top_k: 返回的最大图片数量
        - filters: 额外的 metadata 过滤条件（例如按用户/标签/时间等）
        """
        if top_k <= 0:
            return []

        emb = self._embed_image_bytes(target_image_bytes)

        where: Dict[str, Any] = {"type": "image"}
        if filters:
            # 用户传入的过滤条件优先级更高，可以覆盖默认字段
            where.update(filters)

        collection = getattr(self.vector_store, "_collection", None)
        if collection is None:
            # 理论上不会发生，作为兜底使用 similarity_search_by_vector
            logger.warning("Chroma collection 对象不存在，回退到 similarity_search_by_vector")
            docs = self.vector_store.similarity_search_by_vector(emb, k=top_k)
            refs: List[ImageRef] = []
            for doc in docs:
                meta = dict(doc.metadata or {})
                md5_hex = str(meta.get("md5", ""))
                source = str(meta.get("source", ""))
                path = self._resolve_image_path(md5_hex, source)
                refs.append(
                    ImageRef(
                        md5=md5_hex,
                        source=source,
                        path=path,
                        score=None,
                        metadata=meta,
                    )
                )
            return refs

        res = collection.query(
            query_embeddings=[emb],
            n_results=top_k,
            where=where,
            include=["metadatas", "distances"],
        )

        metadatas = res.get("metadatas") or [[]]
        distances = res.get("distances") or [[]]

        result: List[ImageRef] = []
        if not metadatas or not metadatas[0]:
            return result

        for meta, dist in zip(metadatas[0], distances[0] if distances and distances[0] else []):
            meta = dict(meta or {})
            md5_hex = str(meta.get("md5", ""))
            source = str(meta.get("source", ""))
            path = self._resolve_image_path(md5_hex, source)
            score: Optional[float] = None
            try:
                if dist is not None:
                    score = float(dist)
            except Exception:
                score = None

            result.append(
                ImageRef(
                    md5=md5_hex,
                    source=source,
                    path=path,
                    score=score,
                    metadata=meta,
                )
            )

        return result


__all__ = ["ImageRef", "ImageRetrievalService"]

