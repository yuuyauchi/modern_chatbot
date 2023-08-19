import json
import re
from typing import List

import requests
from googleapiclient.discovery import build
from lxml import html
from requests_html import HTMLSession
from tqdm import tqdm

CUSTOM_SEARCH_ENGINE_ID = "ここにSearch engine IDを入力"
API_KEY = "ここにCustom Search APIのAPIキーを入力"


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
        file_path = re.search(r"(?<=\')(.*?)(?=\')", target).group()
        file_path_list.append(f"{domain}{file_path}")

    player_list = []

    session = HTMLSession()
    for file_path in tqdm(file_path_list):
        r = session.get(file_path)
        try:
            xpath_elements = r.html.xpath("//table[@class='player-detail']//td[@class='player-detail-name']")
        except Exception as e:
            print(e)
            continue
        for xpath_element in xpath_elements:
            player_list.append(xpath_element.text)
    dic = {
        "player": list(set(player_list))
    }

    with open("data.json", "w") as json_file:
        json.dump(dic, json_file, ensure_ascii=False)
    return
