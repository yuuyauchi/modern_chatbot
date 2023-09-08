import json
import re
from typing import List, Any

import requests
from googleapiclient.discovery import build
from lxml import html
from requests_html import HTMLSession
from tqdm import tqdm
import os
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import spacy
from llama_index import SimpleWebPageReader, LLMPredictor, GPTVectorStoreIndex, ServiceContext
from langchain.chat_models import ChatOpenAI

from llama_index import SimpleWebPageReader
from os.path import join, dirname
from dotenv import load_dotenv
import openai
import sys
from llama_index.schema import Document
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser import SimpleNodeParser
from llama_index import SimpleDirectoryReader

from dataclasses import dataclass
from typing import Callable, List, Optional

from llama_index.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from llama_index.bridge.langchain import TextSplitter

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.utils import globals_helper
from llama_index.evaluation import DatasetGenerator, ResponseEvaluator

import tinysegmenter
segmenter = tinysegmenter.TinySegmenter()

class OriginalTextSplitter(TokenTextSplitter):
    def __init__(
        self,
        separator: str = " ",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        tokenizer: Optional[Callable] = None,
        backup_separators: str = "\n",
        callback_manager: Optional[CallbackManager] = None,
    ):
        super().__init__(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=tokenizer,
            backup_separators=backup_separators,
            callback_manager=callback_manager,
        )
        self._separator = separator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self.tokenizer = tokenizer or globals_helper.tokenizer
        self._backup_separators = backup_separators
        self.callback_manager = callback_manager or CallbackManager([])

    def _preprocess_splits(self, splits: List[str], chunk_size: int) -> List[str]:
        """Process splits.

        Specifically search for tokens that are too large for chunk size,
        and see if we can separate those tokens more
        (via backup separators if specified, or force chunking).

        """
        segmenter = tinysegmenter.TinySegmenter()
        new_splits = []
        
        for split in splits:
            new_splits.extend(segmenter.tokenize(split))

        new_splits = list(set(new_splits))
        # import pdb;pdb.set_trace()
        return new_splits

class CustomWebPageReader(SimpleWebPageReader):
    def __init__(self, html_to_text: bool = False) -> None:
        super().__init__(html_to_text=html_to_text)
        self.html_to_text = html_to_text
    
    def load_data(self, urls: List[str]) -> List[Document]:
        """Load data from the input directory.

        Args:
            urls (List[str]): List of URLs to scrape.

        Returns:
            List[Document]: List of documents.

        """
        if not isinstance(urls, list):
            raise ValueError("urls must be a list of strings.")
        documents = []
        for url in urls:
            response = requests.get(url, headers=None).text
            if self._html_to_text:
                import html2text
                response = html2text.html2text(response)
            documents.append(Document(text=response))

        return documents

def display_eval_df(query: str, response: Response, eval_result: str) -> None:
    eval_df = pd.DataFrame(
        {
            "Query": query,
            "Response": str(response),
            "Source": response.source_nodes[0].node.get_content()[:1000] + "...",
            "Evaluation Result": eval_result,
        },
        index=[0],
    )
    eval_df = eval_df.style.set_properties(
        **{
            "inline-size": "600px",
            "overflow-wrap": "break-word",
        },
        subset=["Response", "Source"]
    )
    display(eval_df)

# TODO 外部から読み込む
load_dotenv(verbose=True)
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
openai.api_key = os.environ["OPENAI_API_KEY"]

# TODO データ読み込み
documents = SimpleDirectoryReader(input_dir="./data").load_data()
# documents = CustomWebPageReader(html_to_text=True).load_data(urls)

# TODO データ前処理
text_splitter = OriginalTextSplitter(
    separator="。",
    backup_separators="[\n、（）「」]",
    chunk_size=100, 
    chunk_overlap=20
    )

node_parser = SimpleNodeParser(
  text_splitter=text_splitter,
)

llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    node_parser=node_parser
)
evaluator = ResponseEvaluator(service_context=service_context)
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

query = "2023年３月にリオネルメッシはどのクラブに所属していますか？"
# query = "Which football club signed a contract with Messi in 2023"
index = index.as_query_engine(service_context=service_context)
response = index.query(query)
print(response)
import pdb;pdb.set_trace()
data_generator = DatasetGenerator.from_documents(documents)
eval_questions = data_generator.generate_questions_from_nodes()
eval_result = evaluator.evaluate(response)
print(str(eval_result))
index.set_index_id("vector_index")
index.storage_context.persist('storage')
