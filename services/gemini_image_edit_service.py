import base64
import os
from dataclasses import dataclass
from typing import Any, Sequence

from utils.logger_handler import logger


@dataclass
class ImageEditOptions:
    """
    Gemini 图像编辑的一些可调参数。

    - model_id: 使用的模型标识，默认 gemini-3-pro-image-preview（NanoBanana 图像模型）
    - max_ref_images: 单次调用允许的参考图片最大数量
    - mime_type: 目标图与参考图的 MIME 类型（通常为 image/png 或 image/jpeg）
    """

    model_id: str = "gemini-3-pro-image-preview"
    max_ref_images: int = 6
    mime_type: str = "image/png"


class GeminiImageEditService:
    """
    封装对 Gemini / NanoBanana 图像模型的调用逻辑。

    使用方式：
        service = GeminiImageEditService()
        result_bytes = service.edit_image_with_refs(
            instruction=\"保持人物不变，背景改成白色证件照\",
            target_image_bytes=target_bytes,
            ref_image_bytes_list=[...],
        )
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_id: str = "gemini-3-pro-image-preview",
        max_ref_images: int = 6,
        mime_type: str = "image/png",
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or ""
        )
        if not self.api_key:
            raise RuntimeError("未检测到 GEMINI_API_KEY 或 GOOGLE_API_KEY，无法调用 Gemini 图像模型")

        self.model_id = model_id
        self.max_ref_images = max_ref_images
        self.mime_type = mime_type

        try:
            import google.generativeai as genai  # type: ignore
        except Exception as e:  # pragma: no cover - 依赖缺失时的显式错误
            raise RuntimeError(
                "缺少 google-generativeai 依赖，请先安装：pip install google-generativeai"
            ) from e

        genai.configure(api_key=self.api_key)
        self._genai = genai
        self._model = genai.GenerativeModel(self.model_id)

    def _make_image_part(self, image_bytes: bytes) -> Any:
        """
        将图片二进制包装成 Gemini 可识别的 inline_data 结构。
        使用 dict 形式，兼容 SDK 与 HTTP 方式。
        """
        if not image_bytes:
            raise ValueError("image_bytes 不能为空")

        data_b64 = base64.b64encode(image_bytes).decode("ascii")
        return {
            "inline_data": {
                "mime_type": self.mime_type,
                "data": data_b64,
            }
        }

    def edit_image_with_refs(
        self,
        instruction: str,
        target_image_bytes: bytes,
        ref_image_bytes_list: Sequence[bytes],
    ) -> bytes:
        """
        使用 Gemini 图像模型对目标图进行编辑，并利用若干参考图片作为风格/内容参考。

        返回：生成后图片的二进制内容（通常为 PNG / JPEG）。
        """
        if not instruction.strip():
            raise ValueError("instruction 不能为空")
        if not target_image_bytes:
            raise ValueError("target_image_bytes 不能为空")

        # 截断参考图数量，避免超出模型限制
        refs = list(ref_image_bytes_list or [])[: self.max_ref_images]

        parts: list[Any] = [{"text": instruction}]
        parts.append(self._make_image_part(target_image_bytes))
        for b in refs:
            parts.append(self._make_image_part(b))

        try:
            response = self._model.generate_content(parts)
        except Exception as e:
            logger.error("调用 Gemini 图像模型失败：%s: %s", type(e).__name__, e)
            raise

        return self._extract_image_bytes_from_response(response)

    def _extract_image_bytes_from_response(self, response: Any) -> bytes:
        """
        从 Gemini 返回中解析图片结果。

        兼容几种常见结构（候选 & parts & inline_data.data）：
            response.candidates[0].content.parts[*].inline_data.data
        """
        # 尝试属性访问
        candidates = getattr(response, "candidates", None)
        if not candidates and isinstance(response, dict):
            candidates = response.get("candidates")

        if not candidates:
            raise ValueError("Gemini 返回中不存在 candidates，无法解析图片结果")

        for cand in candidates:
            content = getattr(cand, "content", None)
            if content is None and isinstance(cand, dict):
                content = cand.get("content")
            if not content:
                continue

            parts = getattr(content, "parts", None)
            if parts is None and isinstance(content, dict):
                parts = content.get("parts")
            if not parts:
                continue

            for part in parts:
                inline = (
                    getattr(part, "inline_data", None)
                    or getattr(part, "inlineData", None)
                )
                if inline is None and isinstance(part, dict):
                    inline = part.get("inline_data") or part.get("inlineData")
                if not inline:
                    continue

                data_b64 = getattr(inline, "data", None)
                if data_b64 is None and isinstance(inline, dict):
                    data_b64 = inline.get("data")
                if not data_b64:
                    continue

                try:
                    return base64.b64decode(data_b64)
                except Exception:
                    continue

        raise ValueError("未能从 Gemini 返回中解析到图片数据")


__all__ = ["ImageEditOptions", "GeminiImageEditService"]

