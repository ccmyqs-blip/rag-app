import io

import streamlit as st

st.set_page_config(page_title="图像 RAG 编辑", layout="wide")

st.title("图像 RAG 编辑（相似图片检索 + NanoBanana/Gemini）")
st.caption(
    "上传一张待修改图片，系统会在已入库的图片向量数据库中检索相似参考图，"
    "并将这些参考图与目标图一起交给 Gemini/NanoBanana 图像模型进行编辑。"
)
st.divider()

# 惰性初始化流水线；国内无外网或缺少依赖时优雅降级，不报错崩溃
if "image_rag_pipeline" not in st.session_state:
    try:
        from services.image_rag_edit_pipeline import ImageRagEditPipeline
        st.session_state["image_rag_pipeline"] = ImageRagEditPipeline()
        st.session_state["image_rag_edit_error"] = None
    except Exception as e:
        st.session_state["image_rag_pipeline"] = None
        err_msg = str(e).strip()
        if "google-generativeai" in err_msg or "GEMINI_API_KEY" in err_msg or "GOOGLE_API_KEY" in err_msg:
            st.session_state["image_rag_edit_error"] = (
                "图像编辑功能需要外网环境（Google Gemini API）及 GEMINI_API_KEY。"
                "当前环境不可用时，本页仅做说明；请在有外网或代理的机器上使用，或配置好 Key 后重建镜像。"
            )
        else:
            st.session_state["image_rag_edit_error"] = f"图像编辑服务不可用：{err_msg}"

if st.session_state.get("image_rag_edit_error"):
    st.warning(st.session_state["image_rag_edit_error"])
    st.info("您仍可在「RAG服务」页面上传与检索图片；图像 RAG 编辑需 Gemini 可用后再用。")
    st.stop()

pipeline = st.session_state["image_rag_pipeline"]

with st.sidebar:
    st.header("检索与生成参数")
    top_k = st.slider("参考图片数量 (top_k)", min_value=1, max_value=12, value=6, step=1)
    st.caption("实际传给模型的参考图可能会因模型上限被进一步截断。")

st.subheader("1. 上传待修改图片")
target_file = st.file_uploader(
    "选择一张图片（PNG/JPG/JPEG）",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
)

if target_file is not None:
    st.image(target_file, caption="待修改图片预览", use_container_width=True)

st.subheader("2. 输入编辑指令")
instruction = st.text_area(
    "指令",
    placeholder="例如：保持人脸不变，把背景改成白色证件照风格，光线柔和自然。",
)

st.subheader("3. 执行图像 RAG 编辑")
run_edit = st.button(
    "检索相似图片并生成新图",
    type="primary",
    use_container_width=True,
    disabled=not bool(target_file and instruction.strip()),
)

if run_edit and target_file is not None and instruction.strip():
    target_bytes = target_file.getvalue()

    try:
        with st.spinner("检索相似图片并调用图像模型中，请稍候..."):
            result = pipeline.run(
                target_image_bytes=target_bytes,
                instruction=instruction,
                top_k=top_k,
                filters=None,
            )
    except Exception as e:
        st.error(f"图像 RAG 编辑失败：{type(e).__name__}: {e}")
    else:
        st.success("生成完成")

        # 展示参考图片
        if result.ref_images:
            st.markdown("#### 检索到的参考图片")
            cols = st.columns(min(4, len(result.ref_images)))
            for idx, ref in enumerate(result.ref_images):
                with cols[idx % len(cols)]:
                    try:
                        with open(ref.path, "rb") as f:
                            img_bytes = f.read()
                        st.image(
                            img_bytes,
                            caption=f"{ref.source} (score={ref.score:.4f if ref.score is not None else 'N/A'})",
                            use_container_width=True,
                        )
                    except Exception:
                        st.caption(f"{ref.source}（预览失败）")
        else:
            st.info("未在向量库中检索到参考图片，本次仅基于目标图和指令生成。")

        st.markdown("#### 生成结果")
        st.image(
            result.edited_image_bytes,
            caption="编辑后的图片",
            use_container_width=True,
        )

        # 下载按钮
        st.download_button(
            "下载生成图片",
            data=io.BytesIO(result.edited_image_bytes),
            file_name="edited_image.png",
            mime="image/png",
            use_container_width=True,
        )

