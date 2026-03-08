import base64
import hashlib
import json
import os
from datetime import datetime
from typing import Any


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_client_ip() -> str:
    """
    Best-effort client IP detection for Streamlit.
    - If behind proxy, prefers X-Forwarded-For / X-Real-IP.
    - Falls back to empty string when not available.
    """
    try:
        import streamlit as st

        # Streamlit public API (recommended).
        headers = {}
        ctx = getattr(st, "context", None)
        if ctx is not None:
            headers = getattr(ctx, "headers", None) or {}

        xff = headers.get("X-Forwarded-For") or headers.get("x-forwarded-for")
        if xff:
            # XFF can be a comma-separated chain; first one is the client.
            return str(xff).split(",")[0].strip()
        xri = headers.get("X-Real-Ip") or headers.get("X-Real-IP") or headers.get("x-real-ip")
        if xri:
            return str(xri).strip()
    except Exception:
        pass

    return ""


def user_id_from_ip(ip: str) -> str:
    normalized = (ip or "").strip()
    if not normalized:
        normalized = "unknown"
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()[:16]


def _encode_images(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for img in images or []:
        raw = img.get("bytes") or b""
        if isinstance(raw, str):
            # Already encoded
            b64 = raw
        else:
            b64 = base64.b64encode(raw).decode("ascii")
        out.append(
            {
                "name": img.get("name"),
                "mime": img.get("mime"),
                "bytes_b64": b64,
            }
        )
    return out


def _decode_images(images: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for img in images or []:
        b64 = img.get("bytes_b64")
        raw = b""
        if isinstance(b64, str) and b64:
            try:
                raw = base64.b64decode(b64.encode("ascii"), validate=False)
            except Exception:
                raw = b""
        out.append(
            {
                "name": img.get("name"),
                "mime": img.get("mime"),
                "bytes": raw,
            }
        )
    return out


def _normalize_messages_for_save(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for msg in messages or []:
        normalized.append(
            {
                "role": msg.get("role"),
                "content": msg.get("content"),
                "images": _encode_images(msg.get("images") or []),
            }
        )
    return normalized


def _normalize_messages_for_load(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for msg in messages or []:
        normalized.append(
            {
                "role": msg.get("role"),
                "content": msg.get("content"),
                "images": _decode_images(msg.get("images") or []),
            }
        )
    return normalized


def _atomic_write_json(path: str, payload: dict[str, Any]) -> None:
    directory = os.path.dirname(path)
    _safe_mkdir(directory)

    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def _store_root(project_root: str) -> str:
    return os.path.join(project_root, "data", "users")


def user_file_path(project_root: str, user_id: str) -> str:
    return os.path.join(_store_root(project_root), f"{user_id}.json")


def load_user_messages(project_root: str, user_id: str) -> list[dict[str, Any]]:
    path = user_file_path(project_root, user_id)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return _normalize_messages_for_load(payload.get("messages") or [])
    except Exception:
        return []


def save_user_messages(project_root: str, user_id: str, ip: str, messages: list[dict[str, Any]]) -> None:
    path = user_file_path(project_root, user_id)
    payload = {
        "user_id": user_id,
        "ip": ip,
        "updated_at": _now_iso(),
        "messages": _normalize_messages_for_save(messages),
    }
    if not os.path.exists(path):
        payload["created_at"] = payload["updated_at"]
    _atomic_write_json(path, payload)


def delete_user_history(project_root: str, user_id: str) -> None:
    path = user_file_path(project_root, user_id)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass

