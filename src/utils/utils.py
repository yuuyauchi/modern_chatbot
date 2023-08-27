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

# os.environ["OPENAI_API_KEY"] = "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"

# os.environ.get("OPENAI_API_KEY")

CUSTOM_SEARCH_ENGINE_ID = "a49acfd5c69f54121"
API_KEY = "AIzaSyAs3rf-wkDlb7OcUWeLt9ZHjZ2wJfm70H8"


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
"""
urls = get_input("data.json")
loader = UnstructuredURLLoader(urls=urls)
# print(loader.load())

class JapaneseCharacterTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, **kwargs: Any):
        separators = ["\n\n", "\n", "。", "、", " ", ""]
        super().__init__(separators=separators, **kwargs)

text_splitter = JapaneseCharacterTextSplitter(
)

index = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=OpenAIEmbeddings(),
    text_splitter=text_splitter,
).from_loaders([loader])
"""

"""
query = "2023年の和田育の活躍を教えてください"
nlp = spacy.load('ja_ginza_electra')
doc = nlp(query)
query = [ent.text for ent in doc.ents]
urls = get_search_engine_result(query)
"""
load_dotenv(verbose=True)
dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)
openai.api_key = os.environ["OPENAI_API_KEY"]

# urls = get_input("data.json")
args = sys.argv
query = args[1]
nlp = spacy.load('ja_ginza_electra')
doc = nlp(query)
query = [ent.text for ent in doc.ents]
print(query)
urls = get_search_engine_result(query)
documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"))
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)
index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)
index.set_index_id("vector_index")
index.storage_context.persist('storage')
