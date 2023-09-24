from utils.models.langchain_model import LangChainChatBot

# from utils.models.llamaindex_model import LlamaindexChatBot


def main():
    model = LangChainChatBot()
    model.read()
    model.tokenize()
    model.train()
    # model.evaluate("llama_index_result.csv")


if __name__ == "__main__":
    main()
