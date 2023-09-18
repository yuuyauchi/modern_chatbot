from utils.models.llamaindex_model import LlamaindexChatBot


def main():
    model = LlamaindexChatBot()
    model.read()
    model.tokenize()
    model.train()
    model.evaluate("output/llama_index_result.csv")


if __name__ == "__main__":
    main()
