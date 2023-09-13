from typing import Callable, List, Optional

import tinysegmenter
from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks.base import CallbackManager
from llama_index.constants import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE
from llama_index.text_splitter.types import MetadataAwareTextSplitter
from llama_index.text_splitter.utils import split_by_char, split_by_sep
from llama_index.utils import globals_helper

DEFAULT_METADATA_FORMAT_LEN = 2


class TinySegmenterTextSplitter(MetadataAwareTextSplitter):
    """Implementation of splitting text that looks at word tokens."""

    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE, description="The token chunk size for each chunk."
    )
    chunk_overlap: int = Field(
        default=DEFAULT_CHUNK_OVERLAP,
        description="The token overlap of each chunk when splitting.",
    )
    separator: str = Field(
        default=" ", description="Default separator for splitting into words"
    )
    backup_separators: List = Field(
        default_factory=list, description="Additional separators for splitting."
    )
    callback_manager: CallbackManager = Field(
        default_factory=CallbackManager, exclude=True
    )
    tokenizer: Callable = Field(
        default_factory=globals_helper.tokenizer,  # type: ignore
        description="Tokenizer for splitting words into tokens.",
        exclude=True,
    )

    _split_fns: List[Callable] = PrivateAttr()

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        tokenizer: Optional[Callable] = None,
        callback_manager: Optional[CallbackManager] = None,
        separator: str = " ",
        backup_separators: Optional[List[str]] = ["\n"],
    ):
        """Initialize with parameters."""
        if chunk_overlap > chunk_size:
            raise ValueError(
                f"Got a larger chunk overlap ({chunk_overlap}) than chunk size "
                f"({chunk_size}), should be smaller."
            )
        callback_manager = callback_manager or CallbackManager([])
        tokenizer = tokenizer or globals_helper.tokenizer

        all_seps = [separator] + (backup_separators or [])
        self._split_fns = [split_by_sep(sep) for sep in all_seps] + [split_by_char()]

        super().__init__(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            backup_separators=backup_separators,
            callback_manager=callback_manager,
            tokenizer=tokenizer,
        )
        self.tokenizer = tinysegmenter.TinySegmenter()

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "TinySegmenterTextSplitter"

    def split_text_metadata_aware(self, text: str, metadata_str: str) -> List[str]:
        """Split text into chunks, reserving space required for metadata str."""
        metadata_len = (
            len(self.tokenizer.tokenize(metadata_str)) + DEFAULT_METADATA_FORMAT_LEN
        )
        effective_chunk_size = self.chunk_size - metadata_len
        if effective_chunk_size <= 0:
            raise ValueError(
                f"Metadata length ({metadata_len}) is longer than chunk size "
                f"({self.chunk_size}). Consider increasing the chunk size or "
                "decreasing the size of your metadata to avoid this."
            )
        elif effective_chunk_size < 50:
            print(
                f"Metadata length ({metadata_len}) is close to chunk size "
                f"({self.chunk_size}). Resulting chunks are less than 50 tokens. "
                "Consider increasing the chunk size or decreasing the size of "
                "your metadata to avoid this.",
                flush=True,
            )

        return self.split_text(text)

    def split_text(self, text: str) -> List[str]:
        return list(set(self.tokenizer.tokenize(text)))
