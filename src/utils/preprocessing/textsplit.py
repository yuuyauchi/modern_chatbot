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

import tinysegmenter


@dataclass
class TextSplit:
    """Text split with overlap.

    Attributes:
        text_chunk: The text string.
        num_char_overlap: The number of overlapping characters with the previous chunk.
    """

    text_chunk: str
    num_char_overlap: Optional[int] = None

class TinySegmenterTextSplitter(BaseDocumentTransformer, ABC):
    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        callback_manager: Optional[CallbackManager] = None,
    ):
        self._separator = separator
        self.tokenizer = tinysegmenter.TinySegmenter()
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self.callback_manager = callback_manager or CallbackManager([])

    def _reduce_chunk_size(
        self, start_idx: int, cur_idx: int, splits: List[str]
    ) -> int:
        """Reduce the chunk size by reducing cur_idx.

        Return the new cur_idx.

        """
        current_doc_total = len(
            self.tokenizer.tokenize(self._separator.join(splits[start_idx:cur_idx]))
        )
        while current_doc_total > self._chunk_size:
            percent_to_reduce = (
                current_doc_total - self._chunk_size
            ) / current_doc_total
            num_to_reduce = int(percent_to_reduce * (cur_idx - start_idx)) + 1
            cur_idx -= num_to_reduce
            current_doc_total = len(
                self.tokenizer.tokenize(self._separator.join(splits[start_idx:cur_idx]))
            )
        return cur_idx

    def _preprocess_splits(self, splits: List[str], chunk_size: int) -> List[str]:
        new_splits = []
        for split in splits:
            new_splits.extend(self.tokenizer.tokenize(split))
        return new_splits

    def _postprocess_splits(self, docs: List[TextSplit]) -> List[TextSplit]:
        """Post-process splits."""
        # TODO: prune text splits, remove empty spaces
        new_docs = []
        for doc in docs:
            if doc.text_chunk.replace(" ", "") == "":
                continue
            new_docs.append(doc)
        return new_docs

    def split_text(self, text: str, metadata_str: Optional[str] = None) -> List[str]:
        """Split incoming text and return chunks."""
        event_id = self.callback_manager.on_event_start(
            CBEventType.CHUNKING, payload={EventPayload.CHUNKS: text}
        )
        text_splits = self.split_text_with_overlaps(text, metadata_str=metadata_str)
        chunks = [text_split.text_chunk for text_split in text_splits]
        self.callback_manager.on_event_end(
            CBEventType.CHUNKING,
            payload={EventPayload.CHUNKS: chunks},
            event_id=event_id,
        )
        return chunks

    def split_text_with_overlaps(
        self, text: str, metadata_str: Optional[str] = None
    ) -> List[TextSplit]:
        """Split incoming text and return chunks with overlap size."""
        if text == "":
            return []
        event_id = self.callback_manager.on_event_start(
            CBEventType.CHUNKING, payload={EventPayload.CHUNKS: text}
        )

        # NOTE: Consider metadata info str that will be added to the chunk at query time
        #       This reduces the effective chunk size that we can have
        if metadata_str is not None:
            # NOTE: extra 2 newline chars for formatting when prepending in query
            num_extra_tokens = len(self.tokenizer(f"{metadata_str}\n\n")) + 1
            effective_chunk_size = self._chunk_size - num_extra_tokens

            if effective_chunk_size <= 0:
                raise ValueError(
                    "Effective chunk size is non positive after considering metadata"
                )
        else:
            effective_chunk_size = self._chunk_size

        # First we naively split the large input into a bunch of smaller ones.
        splits = text.split(self._separator)
        splits = self._preprocess_splits(splits, effective_chunk_size)
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        docs: List[TextSplit] = []

        start_idx = 0
        cur_idx = 0
        cur_total = 0
        prev_idx = 0  # store the previous end index
        while cur_idx < len(splits):
            cur_token = splits[cur_idx]
            num_cur_tokens = max(len(self.tokenizer.tokenize(cur_token)), 1)
            if num_cur_tokens > effective_chunk_size:
                raise ValueError(
                    "A single term is larger than the allowed chunk size.\n"
                    f"Term size: {num_cur_tokens}\n"
                    f"Chunk size: {self._chunk_size}"
                    f"Effective chunk size: {effective_chunk_size}"
                )
            # If adding token to current_doc would exceed the chunk size:
            # 1. First verify with tokenizer that current_doc
            # 1. Update the docs list
            if cur_total + num_cur_tokens > effective_chunk_size:
                # NOTE: since we use a proxy for counting tokens, we want to
                # run tokenizer across all of current_doc first. If
                # the chunk is too big, then we will reduce text in pieces
                cur_idx = self._reduce_chunk_size(start_idx, cur_idx, splits)
                overlap = 0
                # after first round, check if last chunk ended after this chunk begins
                if prev_idx > 0 and prev_idx > start_idx:
                    overlap = sum([len(splits[i]) for i in range(start_idx, prev_idx)])

                docs.append(
                    TextSplit(self._separator.join(splits[start_idx:cur_idx]), overlap)
                )
                prev_idx = cur_idx
                # 2. Shrink the current_doc (from the front) until it is gets smaller
                # than the overlap size
                # NOTE: because counting tokens individually is an imperfect
                # proxy (but much faster proxy) for the total number of tokens consumed,
                # we need to enforce that start_idx <= cur_idx, otherwise
                # start_idx has a chance of going out of bounds.
                while cur_total > self._chunk_overlap and start_idx < cur_idx:
                    # # call tokenizer on entire overlap
                    # cur_total = self.tokenizer()
                    cur_num_tokens = max(len(self.tokenizer.tokenize(splits[start_idx])), 1)
                    cur_total -= cur_num_tokens
                    start_idx += 1
                # NOTE: This is a hack, make more general
                if start_idx == cur_idx:
                    cur_total = 0
            # Build up the current_doc with term d, and update the total counter with
            # the number of the number of tokens in d, wrt self.tokenizer

            # we reassign cur_token and num_cur_tokens, because cur_idx
            # may have changed
            cur_token = splits[cur_idx]
            num_cur_tokens = max(len(self.tokenizer.tokenize(cur_token)), 1)

            cur_total += num_cur_tokens
            cur_idx += 1
        overlap = 0
        # after first round, check if last chunk ended after this chunk begins
        if prev_idx > start_idx:
            overlap = sum([len(splits[i]) for i in range(start_idx, prev_idx)]) + len(
                range(start_idx, prev_idx)
            )
        docs.append(TextSplit(self._separator.join(splits[start_idx:cur_idx]), overlap))

        # run postprocessing to remove blank spaces
        docs = self._postprocess_splits(docs)
        self.callback_manager.on_event_end(
            CBEventType.CHUNKING,
            payload={EventPayload.CHUNKS: [x.text_chunk for x in docs]},
            event_id=event_id,
        )
        return docs

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            index = -1
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    index = text.find(chunk, index + 1)
                    metadata["start_index"] = index
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
        return documents
    
    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def _join_docs(self, docs: List[str], separator: str) -> Optional[str]:
        text = separator.join(docs)
        text = text.strip()
        if text == "":
            return None
        else:
            return text

    def _merge_splits(self, splits: Iterable[str], separator: str) -> List[str]:
        # We now want to combine these smaller pieces into medium size
        # chunks to send to the LLM.
        separator_len = self._length_function(separator)

        docs = []
        current_doc: List[str] = []
        total = 0
        for d in splits:
            _len = self._length_function(d)
            if (
                total + _len + (separator_len if len(current_doc) > 0 else 0)
                > self._chunk_size
            ):
                if total > self._chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, "
                        f"which is longer than the specified {self._chunk_size}"
                    )
                if len(current_doc) > 0:
                    doc = self._join_docs(current_doc, separator)
                    if doc is not None:
                        docs.append(doc)
                    # Keep on popping if:
                    # - we have a larger chunk than in the chunk overlap
                    # - or if we still have any chunks and the length is long
                    while total > self._chunk_overlap or (
                        total + _len + (separator_len if len(current_doc) > 0 else 0)
                        > self._chunk_size
                        and total > 0
                    ):
                        total -= self._length_function(current_doc[0]) + (
                            separator_len if len(current_doc) > 1 else 0
                        )
                        current_doc = current_doc[1:]
            current_doc.append(d)
            total += _len + (separator_len if len(current_doc) > 1 else 0)
        doc = self._join_docs(current_doc, separator)
        if doc is not None:
            docs.append(doc)
        return docs

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform sequence of documents by splitting them."""
        return self.split_documents(list(documents))

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Asynchronously transform a sequence of documents by splitting them."""
        raise NotImplementedError