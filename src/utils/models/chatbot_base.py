from abc import ABC, abstractmethod


class ChatbotTrainingBase(ABC):
    @abstractmethod
    def read(self):
        pass

    @abstractmethod
    def tokenize(self, document):
        pass

    @abstractmethod
    def train(self, documents, node_parser):
        pass

    @abstractmethod
    def evaluate(self):
        pass
