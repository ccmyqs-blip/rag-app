from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel, ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[BaseChatModel]:
        # 优先使用配置中的模型名；云端旧版本不支持 qwen3 时自动回退
        target_model = rag_conf.get("chat_model_name", "qwen-turbo")
        try:
            # 兼容不同版本：ChatTongyi 的构造函数接受 model 参数
            return ChatTongyi(model=target_model)
        except Exception as e:
            try:
                return ChatTongyi()
            except Exception as e2:
                return None


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings]:
        target_model = rag_conf.get("embedding_model_name", "text-embedding-v1")
        try:
            return DashScopeEmbeddings(model=target_model)
        except Exception:
            try:
                return DashScopeEmbeddings()
            except Exception:
                return None

chat_model = ChatModelFactory().generator()
embedding_model = EmbeddingsFactory().generator()