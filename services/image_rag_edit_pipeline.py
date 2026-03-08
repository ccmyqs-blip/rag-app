from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from rag.image_retrieval_service import ImageRef, ImageRetrievalService
from services.gemini_image_edit_service import GeminiImageEditService
from utils.logger_handler import logger


@dataclass
class ImageEditResult:
    """
    图像 RAG 编辑流水线的输出结果。

    - edited_image_bytes: 生成后的图片二进制内容
    - ref_images: 本次检索并实际参与的参考图片列表
    """

    edited_image_bytes: bytes
    ref_images: List[ImageRef]


class ImageRagEditPipeline:
    """
    将“图片向量检索 + Gemini 图像编辑”串联起来的一条高层流水线。

    典型调用：
        pipeline = ImageRagEditPipeline()
        result = pipeline.run(
            target_image_bytes=target_bytes,
            instruction=\"保持人物不变，背景改成白色证件照\",
            top_k=6,
        )
    """

    def __init__(
        self,
        retrieval_service: Optional[ImageRetrievalService] = None,
        edit_service: Optional[GeminiImageEditService] = None,
    ) -> None:
        self.retrieval_service = retrieval_service or ImageRetrievalService()
        self.edit_service = edit_service or GeminiImageEditService()

    def run(
        self,
        target_image_bytes: bytes,
        instruction: str,
        *,
        top_k: int = 6,
        filters: Optional[Dict[str, Any]] = None,
    ) -> ImageEditResult:
        """
        高层封装：给定目标图片 + 文本指令，自动：
        1. 在向量库中检索相似图片作为参考图；
        2. 调用 Gemini 图像模型生成编辑后的图片。
        """
        if not target_image_bytes:
            raise ValueError("target_image_bytes 不能为空")
        if not instruction.strip():
            raise ValueError("instruction 不能为空")

        # 1. RAG：检索相似参考图
        try:
            refs = self.retrieval_service.retrieve_similar_images(
                target_image_bytes=target_image_bytes,
                top_k=top_k,
                filters=filters or {},
            )
        except Exception as e:
            logger.error("图片相似检索失败：%s: %s", type(e).__name__, e)
            raise

        ref_image_bytes: List[bytes] = []
        for ref in refs:
            if not ref.path:
                continue
            try:
                with open(ref.path, "rb") as f:
                    ref_image_bytes.append(f.read())
            except FileNotFoundError:
                logger.warning("参考图片文件不存在：%s", ref.path)
            except Exception as e:
                logger.warning(
                    "读取参考图片失败：%s (%s): %s", ref.path, type(e).__name__, e
                )

        # 2. 调用 Gemini 图像编辑
        try:
            edited_bytes = self.edit_service.edit_image_with_refs(
                instruction=instruction,
                target_image_bytes=target_image_bytes,
                ref_image_bytes_list=ref_image_bytes,
            )
        except Exception as e:
            logger.error("图像编辑流水线调用 Gemini 失败：%s: %s", type(e).__name__, e)
            raise

        return ImageEditResult(edited_image_bytes=edited_bytes, ref_images=refs)


__all__ = ["ImageEditResult", "ImageRagEditPipeline"]

