from typing import List

import requests
from llama_index import SimpleWebPageReader

# from langchain.docstore.document import Document
from llama_index.schema import Document


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
