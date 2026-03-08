from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from model.factory import chat_model
from utils.prompt_loader import load_rag_prompts


class RagSummarizeRunner:
    """
    从传入的 retriever 执行检索，并用现有 RAG 提示词 + chat_model 做总结。
    用于页面/服务层快速拼装，不依赖固定向量库实现。
    """

    def __init__(self, retriever):
        self.retriever = retriever
        self.prompt_template = PromptTemplate.from_template(load_rag_prompts())
        self.chain = self.prompt_template | chat_model | StrOutputParser()

    def rag_summarize(self, query: str) -> str:
        docs: list[Document] = self.retriever.invoke(query)
        context = ""
        for i, doc in enumerate(docs, start=1):
            context += f"[参考资料{i}]:参考资料：{doc.page_content} | 参考元数据：{doc.metadata}\n"
        return self.chain.invoke({"input": query, "context": context})

