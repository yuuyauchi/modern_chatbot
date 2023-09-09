from utils.utils import setting
import openai
import pdb
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
from utils.preprocessing.textsplit import TinySegmenterTextSplitter
from utils.models.chatbot_base import ChatbotTrainingBase
from utils.models.llamaindex_model import LlamaindexChatBot
# import pdb;pdb.set_trace()
model = LlamaindexChatBot()
model.read()
model.tokenize()
model.train()
model.evaluate("llama_index_result.csv")
# model.index.query(query)

# if __name__ == "__main__":
#     main()