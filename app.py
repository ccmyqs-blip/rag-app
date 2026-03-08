import json
import os
from datetime import datetime
from typing import Any

import streamlit as st

from agent.react_agent import ReactAgent
from utils.path_tool import get_project_root
from utils.user_store import (
    delete_user_history,
    get_client_ip,
    load_user_messages,
    save_user_messages,
    user_id_from_ip,
)


st.set_page_config(
    page_title="通用AI交互界面",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _init_state() -> None:
    project_root = get_project_root()
    ip = get_client_ip()
    user_id = user_id_from_ip(ip)

    if "project_root" not in st.session_state:
        st.session_state["project_root"] = project_root
    if "user_ip" not in st.session_state:
        st.session_state["user_ip"] = ip
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = user_id

    if "agent" not in st.session_state:
        st.session_state["agent"] = ReactAgent()
    if "messages" not in st.session_state:
        st.session_state["messages"] = load_user_messages(project_root, user_id)
    if "pending_images" not in st.session_state:
        st.session_state["pending_images"] = []
    if "app_title" not in st.session_state:
        st.session_state["app_title"] = "个人AI助理"
    if "avatar_user_path" not in st.session_state:
        st.session_state["avatar_user_path"] = ""
    if "avatar_ai_path" not in st.session_state:
        st.session_state["avatar_ai_path"] = ""

    # 头像文件默认从磁盘加载（开箱即用，重启后也能保留）
    avatar_dir = os.path.join(project_root, "data", "ui", "avatars")
    os.makedirs(avatar_dir, exist_ok=True)
    user_avatar = os.path.join(avatar_dir, "user.png")
    ai_avatar = os.path.join(avatar_dir, "assistant.png")
    if not st.session_state["avatar_user_path"] and os.path.exists(user_avatar):
        st.session_state["avatar_user_path"] = user_avatar
    if not st.session_state["avatar_ai_path"] and os.path.exists(ai_avatar):
        st.session_state["avatar_ai_path"] = ai_avatar


def _get_avatar_for_role(role: str) -> str | None:
    if role == "user":
        return st.session_state.get("avatar_user_path") or None
    if role == "assistant":
        return st.session_state.get("avatar_ai_path") or None
    return None


def _chat_message(role: str):
    """
    兼容不同 Streamlit 版本：
    - 新版本支持 st.chat_message(role, avatar=...)
    - 老版本不支持 avatar 参数，自动降级
    """
    avatar = _get_avatar_for_role(role)
    try:
        if avatar:
            return st.chat_message(role, avatar=avatar)
        return st.chat_message(role)
    except TypeError:
        return st.chat_message(role)


def _render_message(msg: dict[str, Any]) -> None:
    role = msg.get("role") or "assistant"
    content = (msg.get("content") or "").strip()

    block = _chat_message(role)
    if content:
        block.write(content)

    images = msg.get("images") or []
    if images:
        cols = block.columns(min(3, len(images)))
        for i, img in enumerate(images):
            with cols[i % len(cols)]:
                st.image(
                    img["bytes"],
                    caption=img.get("name") or "upload",
                    use_container_width=True,
                )


def _capture_stream(generator, cache_list: list[str]):
    for chunk in generator:
        text = chunk or ""
        cache_list.append(text)
        yield text


def _build_prompt(user_text: str, images: list[dict[str, Any]]) -> str:
    if not images:
        return user_text

    image_names = [img.get("name") or "upload" for img in images]
    extra = "\n".join(
        [
            "",
            "（附加信息）用户上传了图片：",
            f"- 数量：{len(images)}",
            f"- 文件名：{', '.join(image_names)}",
            "",
            "注意：当前后端 Agent 可能只支持文本输入；如需我基于图片内容分析，请你补充图片的文字描述（或后续接入支持视觉的模型）。",
        ]
    )
    return f"{user_text}{extra}"


def _export_chat() -> str:
    payload = {
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "user_id": st.session_state.get("user_id"),
        "user_ip": st.session_state.get("user_ip"),
        "messages": st.session_state["messages"],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _set_pending_images(uploads) -> None:
    if not uploads:
        st.session_state["pending_images"] = []
        return

    pending = []
    for f in uploads:
        pending.append({"name": f.name, "mime": f.type, "bytes": f.getvalue()})
    st.session_state["pending_images"] = pending


_init_state()

with st.sidebar:
    st.header("设置")

    st.caption(f"用户ID：{st.session_state.get('user_id')}")
    ip_text = st.session_state.get("user_ip") or "（未能获取IP）"
    st.caption(f"来源IP：{ip_text}")

    st.subheader("界面图标")
    st.caption("上传后会保存到本机，刷新/重启后仍然生效。")

    def _save_avatar(upload, target_path: str) -> None:
        if upload is None:
            return
        data = upload.getvalue()
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "wb") as f:
            f.write(data)

    project_root = st.session_state["project_root"]
    avatar_dir = os.path.join(project_root, "data", "ui", "avatars")
    user_avatar_path = os.path.join(avatar_dir, "user.png")
    ai_avatar_path = os.path.join(avatar_dir, "assistant.png")

    up_user = st.file_uploader(
        "用户头像（PNG/JPG）",
        type=["png", "jpg", "jpeg", "webp"],
        key="avatar_user_upload",
    )
    if up_user is not None:
        _save_avatar(up_user, user_avatar_path)
        st.session_state["avatar_user_path"] = user_avatar_path

    up_ai = st.file_uploader(
        "AI 头像（PNG/JPG）",
        type=["png", "jpg", "jpeg", "webp"],
        key="avatar_ai_upload",
    )
    if up_ai is not None:
        _save_avatar(up_ai, ai_avatar_path)
        st.session_state["avatar_ai_path"] = ai_avatar_path

    c_av1, c_av2 = st.columns(2)
    with c_av1:
        if st.button("重置用户头像", use_container_width=True):
            st.session_state["avatar_user_path"] = ""
            try:
                if os.path.exists(user_avatar_path):
                    os.remove(user_avatar_path)
            except Exception:
                pass
            st.rerun()
    with c_av2:
        if st.button("重置AI头像", use_container_width=True):
            st.session_state["avatar_ai_path"] = ""
            try:
                if os.path.exists(ai_avatar_path):
                    os.remove(ai_avatar_path)
            except Exception:
                pass
            st.rerun()

    st.session_state["app_title"] = st.text_input("标题", value=st.session_state["app_title"])

    c1, c2 = st.columns(2)
    with c1:
        if st.button("清空对话", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["pending_images"] = []
            delete_user_history(st.session_state["project_root"], st.session_state["user_id"])
            st.rerun()
    with c2:
        st.download_button(
            "导出对话(JSON)",
            data=_export_chat(),
            file_name="chat_export.json",
            mime="application/json",
            use_container_width=True,
        )

st.title(st.session_state["app_title"])
st.caption("通用对话界面：支持聊天、流式输出、图片上传预览与导出对话。")
st.divider()

for msg in st.session_state["messages"]:
    _render_message(msg)

if st.session_state["pending_images"]:
    st.caption("待发送图片预览（发送后会自动清空）")
    st.image(
        [img["bytes"] for img in st.session_state["pending_images"]],
        caption=[img.get("name") or "upload" for img in st.session_state["pending_images"]],
        use_container_width=True,
    )

with st.form("chat_input_form", clear_on_submit=True):
    # 兼容不同 Streamlit 版本：部分版本不支持 vertical_alignment 参数
    c_input, c_send, c_upload = st.columns([0.78, 0.14, 0.08])

    with c_input:
        user_text = st.text_input(
            "输入",
            placeholder="输入你的问题（可点击右侧上传图片）",
            label_visibility="collapsed",
        )

    with c_send:
        submitted = st.form_submit_button("发送", use_container_width=True)

    with c_upload:
        # 兼容不同 Streamlit 版本：老版本可能没有 popover
        if hasattr(st, "popover"):
            with st.popover("📷", use_container_width=True):
                st.caption("上传图片（支持多张，随下一条消息发送）")
                uploads = st.file_uploader(
                    "选择图片",
                    type=["png", "jpg", "jpeg", "webp"],
                    accept_multiple_files=True,
                    label_visibility="collapsed",
                )
                if uploads is not None:
                    _set_pending_images(uploads)
                if st.session_state["pending_images"]:
                    st.caption(f"已选择 {len(st.session_state['pending_images'])} 张图片")
        else:
            with st.expander("📷", expanded=False):
                uploads = st.file_uploader(
                    "选择图片（支持多张，随下一条消息发送）",
                    type=["png", "jpg", "jpeg", "webp"],
                    accept_multiple_files=True,
                )
                if uploads is not None:
                    _set_pending_images(uploads)
                if st.session_state["pending_images"]:
                    st.caption(f"已选择 {len(st.session_state['pending_images'])} 张图片")

if submitted and user_text:
    images = list(st.session_state.get("pending_images") or [])

    user_msg = {"role": "user", "content": user_text, "images": images}
    st.session_state["messages"].append(user_msg)
    save_user_messages(
        st.session_state["project_root"],
        st.session_state["user_id"],
        st.session_state.get("user_ip") or "",
        st.session_state["messages"],
    )

    st.session_state["pending_images"] = []

    with st.spinner("思考中..."):
        response_chunks: list[str] = []
        prompt_for_agent = _build_prompt(user_text, images)

        try:
            res_stream = st.session_state["agent"].execute_stream(prompt_for_agent)
            st.chat_message("assistant").write_stream(
                _capture_stream(res_stream, response_chunks)
            )
            assistant_text = "".join(response_chunks).strip()
        except Exception as e:
            assistant_text = f"调用后端 Agent 失败：{type(e).__name__}: {e}"

        st.session_state["messages"].append({"role": "assistant", "content": assistant_text})
        save_user_messages(
            st.session_state["project_root"],
            st.session_state["user_id"],
            st.session_state.get("user_ip") or "",
            st.session_state["messages"],
        )
        st.rerun()