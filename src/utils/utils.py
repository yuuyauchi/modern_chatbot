import json
import os
import re
from os.path import dirname, join
from typing import List

import openai
import pandas as pd
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
openai.api_key = env["OPENAI_API_KEY"]
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
            open(f"{keyword}.txt", "w").write(content)
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


def generate_answer(prompt: str):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )
    answer = response.choices[0].text.strip()
    return answer


def generate_prompt(name: str, description: str):
    # TODO: プロンプロエンジニアリングによる精度改善
    return f"""
                以下の文章は作成してほしい「質問と回答」の具体例です。
                ＊具体例
                Q.BJ・ペンはいつUFCデビューを果たしたか？\n
                A.BJ・ペンは2001年2月23日のUFC 34でUFCデビューを果たした。

                上記の具体例の文章を参考に以下の{name}に関する文章から質問と解のペアをできる限り多く作成してください。
                {description}
            """


def generate_finetuning_input() -> None:
    if os.path.exists("qa.json"):
        qa_dict = json.load(open("qa.json", "r", encoding="utf-8"))
    else:
        qa_dict = {}
    files = os.listdir("data")

    for file in files:
        f = open(f"data/{file}", "r", encoding="UTF-8")
        data = f.read()
        descriptions = data.split("\n\n")
        name = file[:-3]
        if name in qa_dict.keys():
            print("skip:", name)
            continue
        qa_list = []
        print(name)
        for description in descriptions:
            try:
                prompt = generate_prompt(name, description)
                response = generate_answer(prompt)
                qa_list.append(response)
            except Exception:
                # TODO: try except構文ではなくトークン数のif文を用いる。
                print(len(description))
                splitted_text = description.split("\n")
                chunk_size = int(len(splitted_text) / 2)
                first_chunk = "".join(splitted_text[:chunk_size])
                second_chunk = "".join(splitted_text[chunk_size:])
                prompt = generate_prompt(name, first_chunk)
                response = generate_answer(prompt)
                qa_list.append(response)
                prompt = generate_prompt(name, second_chunk)
                response = generate_answer(prompt)
                qa_list.append(response)
                continue
        qa_dict[name] = qa_list
        with open("qa.json", mode="w") as f:
            d = json.dumps(qa_dict, ensure_ascii=False)
            f.write(d)


def get_finetune_template(content, question, answer):
    template = {
        "messages": [
            {"role": "system", "content": content},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }
    return template


def get_finetune_input():
    if os.path.exists("qa.json"):
        qa_dict = json.load(open("qa.json", "r", encoding="utf-8"))
    else:
        qa_dict = {}

    qa_list = []
    dataset = []
    for qa in qa_dict.values():
        qa_list.extend(qa)

    for qa in qa_list:
        qa_str_list = list(filter(lambda string: string != "", qa.split("\n")))
        it = iter(qa_str_list)
        for question, answer in zip(it, it):
            if "Q" not in question or "A" not in answer:
                continue
            content = "あなたは格闘技について最新の情報を知っているチャットボットです。"
            template = get_finetune_template(content, question, answer)
            dataset.append(template)
    df = pd.DataFrame(dataset)
    df.to_json("input_data.jsonl", force_ascii=False, lines=True, orient="records")
    # TODO データ作成箇所とモデル生成箇所の処理を別関数として分ける。
    upload_response = openai.File.create(
        file=open("input_data.jsonl", "rb"), purpose="fine-tune"
    )
    file_id = upload_response.id
    fine_tune_response = openai.FineTuningJob.create(
        training_file=file_id, model="gpt-3.5-turbo"
    )
    print(fine_tune_response)
