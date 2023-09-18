import json
import os
import re
from os.path import dirname, join
from typing import List

import requests
import wikipedia
from dotenv import load_dotenv
from googleapiclient.discovery import build
from lxml import html
from requests_html import HTMLSession
from tqdm import tqdm

wikipedia.set_lang("jp")


def setting():
    load_dotenv(verbose=True)
    dotenv_path = join(dirname(__file__), ".env")
    load_dotenv(dotenv_path)
    return os.environ


env = setting()
API_KEY = env["API_KEY"]
CUSTOM_SEARCH_ENGINE_ID = env["CUSTOM_SEARCH_ENGINE_ID"]


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
    with open(file_name, "r", encoding="utf-8") as json_file:
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


def get_article_data_from_wikipedia(keyword: str) -> None:
    words = wikipedia.search(keyword)
    for word in words:
        if keyword == word:
            content = wikipedia.page(words[0]).content
            open(f"data/{keyword}.txt", "w").write(content)
            break


def get_youtube_video_urls(channel_id: str) -> List[str]:
    API_VER = "v3"
    youtube = build("youtube", API_VER, developerKey=API_KEY)

    search_response = (
        youtube.search()
        .list(
            channelId=channel_id,
            part="snippet",
            order="date",
            maxResults=50,
        )
        .execute()
    )

    url_list = []
    youtube_url = "https://www.youtube.com/watch?v="
    for response in search_response["items"]:
        if response["id"]["kind"] != "youtube#video":
            continue
        video_url = youtube_url + response["id"]["videoId"]
        url_list.append(video_url)
    return url_list
