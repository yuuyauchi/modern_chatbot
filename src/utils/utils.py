import json                                 # 返却された検索結果の読み取りにつかう
from googleapiclient.discovery import build # APIへのアクセスにつかう

# カスタム検索エンジンID
CUSTOM_SEARCH_ENGINE_ID = "25b4f45d834604076"
# API キー
API_KEY = "AIzaSyB0kOllmst-jHSiAFYNqQL4Dqcrhw_FEgw"

def get_search_engine_result(query: str) -> list[str]:
    
    search = build(
    "customsearch", 
    "v1", 
    developerKey = API_KEY
    )
    
    result = search.cse().list(
    q = query,
    cx = CUSTOM_SEARCH_ENGINE_ID,
    lr = 'lang_ja',
    num = 10,
    start = 1
    ).execute()

    urls = []
    for item in result['items']:
        urls.append(item['link'])

    return urls
    