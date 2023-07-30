from googleapiclient.discovery import build

CUSTOM_SEARCH_ENGINE_ID = "ここにSearch engine IDを入力"
API_KEY = "ここにCustom Search APIのAPIキーを入力"

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
