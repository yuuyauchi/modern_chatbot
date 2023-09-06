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


def get_search_engine_result(query: str) -> List[str]:
    search = build("customsearch", "v1", developerKey=API_KEY)

    result = (
        search.cse()
        .list(q=query, cx=CUSTOM_SEARCH_ENGINE_ID, lr="lang_ja", num=10, start=1)
        .execute()
    )

    urls = []
    for item in result["items"]:
        urls.append(item["link"])

    return urls


def get_player_list() -> None:
    domain = "https://web.ultra-soccer.jp"
    url = f"{domain}/profile/teamlist"

    response = requests.get(url)

    tree = html.fromstring(response.content)
    hrefs = tree.xpath("//div[@class='team-list']/table")
    file_path_list = []
    for href in hrefs:
        target = href.attrib["onclick"]
        file_path_obj = re.search(r"(?<=\')(.*?)(?=\')", target)
        if file_path_obj is None:
            continue
        file_path = file_path_obj.group()
        file_path_list.append(f"{domain}{file_path}")

    player_list = []

    session = HTMLSession()
    for file_path in tqdm(file_path_list):
        r = session.get(file_path)
        try:
            xpath_elements = r.html.xpath(
                "//table[@class='player-detail']//td[@class='player-detail-name']"
            )
        except Exception as e:
            print(e)
            continue
        for xpath_element in xpath_elements:
            player_list.append(xpath_element.text)
    dic = {"player": list(set(player_list))}

    with open("data.json", "w") as json_file:
        json.dump(dic, json_file, ensure_ascii=False)
    return

def get_input(file_name: str) -> List[str]:
    url_list = []
    pdf_list = []
    excel_list = []
    output_list = []
    with open(file_name, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    
    for urls in tqdm(data["search_result"].values()):
        url_list.extend(urls)

    for url in url_list:
        if ".pdf" in url:
            pdf_list.append(url)
        elif ".xlsx" in url:
            excel_list.append(url)
        else:
            output_list.append(url)
    return output_list


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
            # link_list = re.findall(r'(?<=href=").*?(?=")', response)
            # src_list = re.findall(r'(?<=src=").*?(?=")', response)
            # title_list = re.findall(r'(?<=title=").*?(?=")', response)
            # for src in src_list:
            #     response = response.replace(src, "")
            
            # for link in link_list:
            #     response = response.replace(link, "")
            # for title in title_list:
            #     response = response.replace(title, "")
            if self._html_to_text:
                import html2text

                response = html2text.html2text(response)
                # response = re.sub(r"\n", "hogehogemaru", response)
                # link_list = re.findall(r"\(.*?\)", response)
                # for link in link_list:
                #     if link.count("(") == link.count(")"):
                #         response = response.replace(link, "")
                # import pdb;pdb.set_trace()
                # response = re.sub("hogehogemaru", r"\n", response)
            documents.append(Document(text=response))

        return documents

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
        new_splits = []
        for split in splits:
            num_cur_tokens = len(self.tokenizer(split))
            if num_cur_tokens <= chunk_size:
                new_splits.append(split)
            else:
                cur_splits = [split]
                cur_split = re.split(self._backup_separators, split)
                # import pdb; pdb.set_trace()
                new_splits.extend(cur_split)

                cur_splits2 = []
                for cur_split in cur_splits:
                    num_cur_tokens = len(self.tokenizer(cur_split))
                    if num_cur_tokens <= chunk_size:
                        cur_splits2.extend([cur_split])
                    else:
                        # split cur_split according to chunk size of the token numbers
                        cur_split_chunks = []
                        end_idx = len(cur_split)
                        while len(self.tokenizer(cur_split[0:end_idx])) > chunk_size:
                            for i in range(1, end_idx):
                                tmp_split = cur_split[0 : end_idx - i]
                                if len(self.tokenizer(tmp_split)) <= chunk_size:
                                    cur_split_chunks.append(tmp_split)
                                    cur_split = cur_split[end_idx - i : end_idx]
                                    end_idx = len(cur_split)
                                    break
                        cur_split_chunks.append(cur_split)
                        cur_splits2.extend(cur_split_chunks)

                new_splits.extend(cur_splits2)
        return new_splits

load_dotenv(verbose=True)
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
openai.api_key = os.environ["OPENAI_API_KEY"]


urls = [
    "https://ja.wikipedia.org/wiki/%E3%83%AA%E3%82%AA%E3%83%8D%E3%83%AB%E3%83%BB%E3%83%A1%E3%83%83%E3%82%B7"
]
documents = SimpleDirectoryReader(input_dir="./data").load_data()
text_splitter = OriginalTextSplitter(
    separator="。",
    backup_separators="[\n、（）「」]",
    chunk_size=100, 
    chunk_overlap=20
    )
# documents = CustomWebPageReader(html_to_text=True).load_data(urls)
node_parser = SimpleNodeParser(
  text_splitter=text_splitter,

)
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(
    llm_predictor=llm_predictor,
    node_parser=node_parser
)
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
import pdb;pdb.set_trace()
query = "2023年３月にリオネルメッシはどのクラブに所属していますか？"
index = index.as_query_engine(service_context=service_context)
answer = index.query(query)
print(answer)
index.set_index_id("vector_index")
index.storage_context.persist('storage')
