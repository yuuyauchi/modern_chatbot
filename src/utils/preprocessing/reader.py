from typing import List

import requests
from llama_hub.youtube_transcript import YoutubeTranscriptReader, is_youtube_video
from llama_index import SimpleWebPageReader
from llama_index.schema import Document
from utils.utils import get_youtube_video_urls


class YoutubeReader:
    def __init__(self):
        pass

    def get_video_from_channel_id(self, channel_id: str):
        self.urls = get_youtube_video_urls(channel_id)

    def check_url(self) -> None:
        for url in self.urls:
            if is_youtube_video(url):
                continue
            else:
                print(url)

    def get_documents(self) -> Document:
        loader = YoutubeTranscriptReader()
        documents = loader.load_data(ytlinks=self.urls, languages=["ja"])
        return documents


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
            if self._html_to_text:
                import html2text

                response = html2text.html2text(response)
            documents.append(Document(text=response))

        return documents
