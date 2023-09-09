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
from time import time
from llama_index import load_index_from_storage
from llama_index import StorageContext
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores import SimpleVectorStore
from llama_index.storage.index_store import SimpleIndexStore

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

    def tokenize(self):
        self.text_splitter = TinySegmenterTextSplitter(
            separator="。",
            chunk_size=100, 
            chunk_overlap=20
        )

        self.node_parser = SimpleNodeParser(
            text_splitter=self.text_splitter,
        )
        llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name=self.model_name))
        self.service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor,
            node_parser=self.node_parser
        )

    def train(self):
        self.index = GPTVectorStoreIndex.from_documents(self.documents, service_context=self.service_context)
        self.query_engine = self.index.as_query_engine(service_context=self.service_context)
        # TODO 以下の処理をどう管理するか
        self.save_question_and_answer("llama_index_result.csv")
        self.save_model()
        
    def save_question_and_answer(self, file_name):
        data_generator = DatasetGenerator.from_documents(self.documents)
        eval_questions = data_generator.generate_questions_from_nodes(50)
        eval_df = pd.DataFrame(
            {
                "query": eval_questions,
            }
        )
        # eval_df["Response"] = eval_df["Query"].apply(lambda query: self.index.query(query))
        eval_df.to_csv(file_name, index=False)

    def evaluate(self, file_name: str):
        df = pd.read_csv(file_name)
        self.text_splitter = TinySegmenterTextSplitter(
            separator="。",
            chunk_size=100, 
            chunk_overlap=20
        )

        self.node_parser = SimpleNodeParser(
            text_splitter=self.text_splitter,
        )
        llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name=self.model_name))
        self.service_context = ServiceContext.from_defaults(
            llm_predictor=llm_predictor,
            node_parser=self.node_parser
        )

        storage_context = StorageContext.from_defaults(
            docstore=SimpleDocumentStore.from_persist_dir(persist_dir=self.model_path),
            vector_store=SimpleVectorStore.from_persist_dir(persist_dir=self.model_path),
            index_store=SimpleIndexStore.from_persist_dir(persist_dir=self.model_path),
        )
        # import pdb;pdb.set_trace()
        vector_store_index = load_index_from_storage(storage_context, service_context=self.service_context)
        self.query_engine = vector_store_index.as_query_engine(service_context=self.service_context)

        response_list = []
        label_list = []
        self.evaluator = ResponseEvaluator(service_context=self.service_context)
        for query in df["query"].values:
            response = self.query_engine.query(query)
            response_list.append(response)
            label = str(self.evaluator.evaluate(response))
            print(label)
            label_list.append(label)
        df["response"] = response_list
        df["label"] = label_list
        df.to_csv(file_name, index=False)

    def save_model(self):
        # self.index.set_index_id("vector_index")
        self.index.storage_context.persist(self.model_path)

