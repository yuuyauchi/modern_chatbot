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
from llama_index import SimpleDirectoryReader, Response

from dataclasses import dataclass
from typing import Callable, List, Optional

from llama_index.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from llama_index.bridge.langchain import TextSplitter

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.utils import globals_helper
from llama_index.evaluation import DatasetGenerator, ResponseEvaluator

import logging
from typing import Callable, List, Optional

# from llama_index.bridge.pydantic import Field, PrivateAttr

from llama_index.callbacks.base import CallbackManager
from llama_index.callbacks.schema import CBEventType, EventPayload
from llama_index.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from llama_index.utils import globals_helper

import copy
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import (
    AbstractSet,
    Any,
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypedDict,
    TypeVar,
    Union,
    cast,
)

from langchain.docstore.document import Document
from langchain.schema import BaseDocumentTransformer
from llama_index.callbacks.base import CallbackManager
from utils.models.chatbot_base import ChatbotTrainingBase
import tinysegmenter
from utils.preprocessing.textsplit import TinySegmenterTextSplitter
from utils.utils import setting
import pandas as pd


segmenter = tinysegmenter.TinySegmenter()

env = setting()
openai.api_key = env["OPENAI_API_KEY"]

@dataclass
class LlamaindexChatBot(ChatbotTrainingBase):
    data_path: str = "/workspaces/modern_chatbot/data"
    model_name: str = "gpt-3.5-turbo"
    model_path: str = "llama_index_storage"

    def read(self):
        self.documents = SimpleDirectoryReader(input_dir=self.data_path).load_data()

    def preprocess(self):
        self.text_splitter = TinySegmenterTextSplitter(
            separator="。",
            chunk_size=100, 
            chunk_overlap=20
        )

        self.node_parser = SimpleNodeParser(
            text_splitter=self.text_splitter,
        )

    def train(self):
        llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name=self.model_name))
        self.service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor,
            node_parser=self.node_parser
        )
        self.index = GPTVectorStoreIndex.from_documents(self.documents, service_context=self.service_context)
        self.index = self.index.as_query_engine(service_context=self.service_context)

    def save_eval_df(self, query: list) -> None:
        eval_df = pd.DataFrame({"Query": query})
        eval_df["Response"] = eval_df["Query"].apply(lambda query: self.index.query(query))
        eval_df["Evaluation Result"] = eval_df["Response"].apply(lambda response: self.evaluator.evaluate(response))
        eval_df["Response"] = eval_df["Response"].astype(str)
        import pdb;pdb.set_trace()
        eval_df.to_csv("result.csv", index=False)

    def evaluate(self):
        data_generator = DatasetGenerator.from_documents(self.documents)
        eval_questions = data_generator.generate_questions_from_nodes()
        eval_df = pd.DataFrame(
            {
                "Query": eval_questions
            }
        )
        eval_df["Response"] = eval_df["Query"].apply(lambda query: self.index.query(query))
        eval_df.to_csv("result.csv", index=False)
        import pdb;pdb.set_trace()
        self.evaluator = ResponseEvaluator(service_context=self.service_context)
        self.save_eval_df(eval_questions)
        import pdb;pdb.set_trace()

    def save_model(self):
        self.index.set_index_id("vector_index")
        self.index.storage_context.persist(self.model_path)


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
