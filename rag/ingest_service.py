import hashlib
import json
import os
from datetime import datetime
from typing import Any, Iterable
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model.factory import embedding_model
from utils.config_handler import chroma_conf
from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from rag.vl_embeddings import Qwen3VLEmbeddings, _extract_embedding_from_dashscope_response


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


class JsonlDedupeStore:
    """
    简单的查重索引：一行一个 JSON 记录，按 md5 去重。
    设计目标：易追加、坏行可跳过、无需一次性加载大 JSON。
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        _ensure_dir(os.path.dirname(file_path))
        if not os.path.exists(file_path):
            open(file_path, "a", encoding="utf-8").close()

    def has(self, md5_hex: str) -> bool:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if obj.get("md5") == md5_hex:
                        return True
        except Exception:
            return False
        return False

    def add(self, record: dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("ingested_at", _now_iso())
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _ocr_image_to_text(image_path: str) -> tuple[str, str]:
    """
    尝试 OCR 图片为文本：
    - 优先 DashScope/Qwen-VL（开箱即用：只需配置 DASHSCOPE_API_KEY）
    - 其次 pytesseract（需要本机安装 Tesseract）
    - 失败时返回占位文本，保证仍可入库（但检索效果有限）
    返回：(text, method)
    """
    # 1) 云端多模态 OCR（推荐）
    api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
    if api_key:
        try:
            import dashscope  # type: ignore

            # 仅设置一次即可；如果外部已设置则不重复覆盖
            if not getattr(dashscope, "api_key", None):
                dashscope.api_key = api_key

            # 尽量用 file URI，避免 base64 过大
            image_uri = Path(image_path).resolve().as_uri()

            # dashscope SDK 的多模态接口在不同版本字段可能略有差异，这里做容错解析
            from dashscope import MultiModalConversation  # type: ignore

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": image_uri},
                        {
                            "text": "请只输出图片中的文字内容（尽量保持原始顺序与换行）。如果没有文字，请输出空字符串。",
                        },
                    ],
                }
            ]
            resp = MultiModalConversation.call(model="qwen3-vl-plus", messages=messages)

            # 解析输出
            text = ""
            try:
                # 常见：resp.output.choices[0].message.content -> list[{"text": "..."}]
                content = resp.output["choices"][0]["message"]["content"]
                if isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and "text" in item:
                            parts.append(item["text"])
                    text = "\n".join([p for p in parts if p]).strip()
                elif isinstance(content, str):
                    text = content.strip()
            except Exception:
                try:
                    content = resp.output.choices[0].message.content  # type: ignore[attr-defined]
                    if isinstance(content, list):
                        parts = []
                        for item in content:
                            if isinstance(item, dict) and "text" in item:
                                parts.append(item["text"])
                        text = "\n".join([p for p in parts if p]).strip()
                    elif isinstance(content, str):
                        text = content.strip()
                except Exception:
                    text = ""

            if text:
                return text, "qwen3-vl-plus"
            return "", "qwen3-vl-plus"
        except Exception as e:
            logger.warning(f"OCR(dashscope)失败：{type(e).__name__}: {e}")

    # 2) 本机 OCR（可选）
    try:
        from PIL import Image  # type: ignore

        try:
            import pytesseract  # type: ignore

            text = pytesseract.image_to_string(Image.open(image_path))
            text = (text or "").strip()
            if text:
                return text, "pytesseract"
        except Exception as e:
            logger.warning(f"OCR(pytesseract)失败：{type(e).__name__}: {e}")
    except Exception as e:
        logger.warning(f"图片解析失败：{type(e).__name__}: {e}")

    # 兜底：占位文本，满足“入库”但不可读图
    return f"[IMAGE] {os.path.basename(image_path)}", "placeholder"


class UploadIngestService:
    def __init__(self, collection_name: str | None = None):
        # 统一用 qwen3-vl-embedding 作为向量空间：支持文本<->图片 检索
        vl_collection = collection_name or f"{chroma_conf['collection_name']}_vl"
        self.vl_embeddings = Qwen3VLEmbeddings(model_name="qwen3-vl-embedding")
        self.vector_store = Chroma(
            collection_name=vl_collection,
            embedding_function=self.vl_embeddings,
            persist_directory=chroma_conf["persist_directory"],
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

        # 上传文件原文存放目录 & 查重索引（不与原 md5.text 冲突）
        self.upload_dir = get_abs_path(os.path.join("data", "uploads"))
        self.dedupe_index = JsonlDedupeStore(get_abs_path(os.path.join("data", "upload_index.jsonl")))
        _ensure_dir(self.upload_dir)
        self.collection_name = vl_collection

    def _save_original(self, data: bytes, original_name: str, md5_hex: str) -> str:
        safe_name = os.path.basename(original_name or "upload")
        target = os.path.join(self.upload_dir, f"{md5_hex}_{safe_name}")
        if not os.path.exists(target):
            with open(target, "wb") as f:
                f.write(data)
        return target

    def _split_and_add(self, docs: Iterable[Document]) -> int:
        docs_list = list(docs)
        if not docs_list:
            return 0
        chunks = self.splitter.split_documents(docs_list)
        if not chunks:
            return 0
        self.vector_store.add_documents(chunks)
        return len(chunks)

    def _add_image_embedding(self, image_path: str, filename: str, md5_hex: str, extra_meta: dict[str, Any]) -> int:
        """
        将图片直接以 qwen3-vl-embedding 的图片向量写入 Chroma（不走文本 embedding_function）。
        返回：新增条目数量（当前为 1）。
        """
        api_key = os.environ.get("DASHSCOPE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("未检测到环境变量 DASHSCOPE_API_KEY，无法对图片生成 embedding")

        from dashscope.embeddings.multimodal_embedding import (  # type: ignore
            MultiModalEmbedding,
            MultiModalEmbeddingItemImage,
        )

        image_uri = Path(image_path).resolve().as_uri()
        resp = MultiModalEmbedding.call(
            model="qwen3-vl-embedding",
            input=[MultiModalEmbeddingItemImage(image=image_uri, factor=1.0)],
            api_key=api_key,
            auto_truncation=True,
        )
        emb = _extract_embedding_from_dashscope_response(resp)

        # 直接写入底层 collection，避免 Chroma 再次对 page_content 做文本 embedding
        doc_id = f"{md5_hex}:image"
        meta = {"source": filename, "md5": md5_hex, "type": "image", **(extra_meta or {})}
        self.vector_store._collection.add(  # type: ignore[attr-defined]
            ids=[doc_id],
            embeddings=[emb],
            documents=[f"[IMAGE] {filename}"],
            metadatas=[meta],
        )
        return 1

    def ingest_upload(self, filename: str, data: bytes) -> dict[str, Any]:
        """
        入库单个上传文件。返回结果 dict（可直接给页面展示）。
        """
        filename = filename or "upload"
        ext = os.path.splitext(filename)[1].lower().lstrip(".")
        md5_hex = md5_bytes(data)

        if self.dedupe_index.has(md5_hex):
            return {
                "filename": filename,
                "ext": ext,
                "md5": md5_hex,
                "status": "skipped",
                "reason": "md5 已存在（查重命中）",
                "chunks_added": 0,
            }

        saved_path = self._save_original(data, filename, md5_hex)

        docs: list[Document] = []
        ocr_method = None
        image_item_added = 0

        try:
            if ext == "txt":
                text = data.decode("utf-8", errors="ignore")
                docs = [Document(page_content=text, metadata={"source": filename, "md5": md5_hex, "type": "txt"})]
            elif ext == "pdf":
                from langchain_community.document_loaders import PyPDFLoader

                docs = PyPDFLoader(saved_path).load()
                for d in docs:
                    d.metadata = {**(d.metadata or {}), "source": filename, "md5": md5_hex, "type": "pdf"}
            elif ext in ("png", "jpg", "jpeg"):
                # 1) 用 VLM 提取文本（用于展示/辅助理解；检索主要依赖图片向量）
                text, ocr_method = _ocr_image_to_text(saved_path)
                # 2) 直接写入图片 embedding
                image_item_added = self._add_image_embedding(
                    saved_path,
                    filename,
                    md5_hex,
                    extra_meta={"ocr": ocr_method, "ocr_text": (text or "")[:2000]},
                )
                docs = []
            else:
                return {
                    "filename": filename,
                    "ext": ext,
                    "md5": md5_hex,
                    "status": "failed",
                    "reason": "不支持的文件类型",
                    "chunks_added": 0,
                }

            chunks_added = self._split_and_add(docs)

            self.dedupe_index.add(
                {
                    "md5": md5_hex,
                    "filename": filename,
                    "ext": ext,
                    "saved_path": saved_path,
                    "chunks_added": chunks_added,
                    "image_items_added": image_item_added,
                    "ocr": ocr_method,
                }
            )

            return {
                "filename": filename,
                "ext": ext,
                "md5": md5_hex,
                "status": "ok",
                "reason": "",
                "chunks_added": chunks_added,
                "image_items_added": image_item_added,
                "ocr": ocr_method,
            }
        except Exception as e:
            logger.error(f"上传入库失败：{filename} | {type(e).__name__}: {e}")
            return {
                "filename": filename,
                "ext": ext,
                "md5": md5_hex,
                "status": "failed",
                "reason": f"{type(e).__name__}: {e}",
                "chunks_added": 0,
                "image_items_added": 0,
            }

