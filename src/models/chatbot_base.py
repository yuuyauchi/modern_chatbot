from abc import ABC, abstractmethod


class ChatbotTrainingBase(ABC):
    @abstractmethod
    def read_data(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass

    @abstractmethod
    def generate_engine(self, documents, node_parser):
        pass
