from dataclasses import dataclass
import openai
import pandas as pd
import tinysegmenter
from langchain.chat_models import ChatOpenAI
from utils.models.chatbot_base import ChatbotTrainingBase
from utils.preprocessing.textsplit import TinySegmenterTextSplitter
from utils.utils import setting
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

segmenter = tinysegmenter.TinySegmenter()

env = setting()
openai.api_key = env["OPENAI_API_KEY"]


@dataclass
class LangChainChatBot(ChatbotTrainingBase):
    data_path: str = "data"
    model_name: str = "gpt-3.5-turbo"
    model_path: str = "storage"

    def read(self):
        self.loader = DirectoryLoader("data1")

    def tokenize(self):
        # TODO 正答の精度を上げるために異なるsplitterも試す。
        self.text_splitter = CharacterTextSplitter(
        separator = "。",
        chunk_size = 1000,
        chunk_overlap = 50,
    )

    def train(self):
        self.index = VectorstoreIndexCreator(
            vectorstore_cls=Chroma,
            embedding=OpenAIEmbeddings(),
            text_splitter = self.text_splitter
        ).from_loaders([self.loader])
        import pdb;pdb.set_trace()
        self.index.query()

    def evaluate(self):
        pass