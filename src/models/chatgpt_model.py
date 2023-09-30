from dataclasses import dataclass


@dataclass
class LangChainChatBot(ChatbotTrainingBase):
    data_path: str = "finetune.json"
    model_name: str = "gpt-3.5-turbo"

    def read_data(self):
        pass

    def preprocess(self):
        pass

    def generate_engine(self):
        upload_response = openai.File.create(file=open("input_data.jsonl", "rb"), purpose="fine-tune")
        file_id = upload_response.id
        fine_tune_response = openai.FineTuningJob.create(training_file=file_id, model="gpt-3.5-turbo")
        print(fine_tune_response)
        with open(self.data_path, 'w') as json_file:
            json.dump(fine_tune_response, json_file)
    