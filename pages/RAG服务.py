import streamlit as st

from rag.ingest_service import UploadIngestService
from rag.rag_runner import RagSummarizeRunner


st.set_page_config(page_title="数据库存储服务", layout="wide")

st.title("数据库存储服务")
st.caption("上传文件入向量数据库（TXT / PDF / PNG / JPG），并基于 MD5 做查重（同一文件不会重复入库）。")
st.divider()

if "ingest_service" not in st.session_state:
    st.session_state["ingest_service"] = UploadIngestService()

ingest: UploadIngestService = st.session_state["ingest_service"]

st.subheader("上传并入库")
uploads = st.file_uploader(
    "选择文件（支持多选）",
    type=["txt", "pdf", "png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if uploads:
    st.write(f"已选择 {len(uploads)} 个文件：")
    for f in uploads:
        st.write(f"- `{f.name}` ({f.size} bytes)")

run_ingest = st.button("开始入库", type="primary", use_container_width=True, disabled=not bool(uploads))

if run_ingest and uploads:
    results = []
    with st.spinner("正在入库..."):
        for f in uploads:
            data = f.getvalue()
            results.append(ingest.ingest_upload(f.name, data))

    st.success("入库完成")
    st.dataframe(results, use_container_width=True, hide_index=True)

    skipped = [r for r in results if r.get("status") == "skipped"]
    failed = [r for r in results if r.get("status") == "failed"]
    ok = [r for r in results if r.get("status") == "ok"]

    st.caption(
        f"统计：成功 {len(ok)}，查重跳过 {len(skipped)}，失败 {len(failed)}。"
        + (
            "（图片：用 qwen3-vl-plus 提取文字，并用 qwen3-vl-embedding 直接写入图片向量）"
            if any(r.get("ext") in ("png", "jpg", "jpeg") for r in results)
            else ""
        )
    )

st.divider()
st.subheader("检索/总结测试（可选）")
st.caption("用于验证向量库已写入并能检索到内容。")

query = st.text_input("问题", placeholder="例如：这个文件讲了什么？")
do_search = st.button("检索并总结", use_container_width=True, disabled=not bool(query))

if do_search and query:
    try:
        if "rag_runner" not in st.session_state:
            retriever = ingest.vector_store.as_retriever(search_kwargs={"k": 3})
            st.session_state["rag_runner"] = RagSummarizeRunner(retriever)
        rag = st.session_state["rag_runner"]
        with st.spinner("检索中..."):
            answer = rag.rag_summarize(query)
        st.write(answer)
    except Exception as e:
        st.error(f"检索失败：{type(e).__name__}: {e}")

