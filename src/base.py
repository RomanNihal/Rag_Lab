from abc import ABC, abstractmethod
from typing import List

class ChatbotInterface(ABC):
    
    @abstractmethod
    def __init__(self, api_key: str, provider: str, model_name: str):
        """
        Initialize with API key, provider, and specific model name.
        """
        self.api_key = api_key
        self.provider = provider
        self.model_name = model_name

    @abstractmethod
    def process_files(self, file_paths: List[str]) -> str:
        pass

    @abstractmethod
    def chat(self, user_query: str) -> str:
        pass
    
    @staticmethod
    @abstractmethod
    def get_name() -> str:
        pass