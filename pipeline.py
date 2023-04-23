import os
import torch

from zoo_params import zoo_params, create_name
from model import gptKILLER
from transformers import pipeline

class Pipeline:
    def __init__(self):
        index = 11
        qa_pipeline = pipeline(
            "question-answering",
            model="Den4ikAI/rubert-tiny-squad",
            tokenizer="Den4ikAI/rubert-tiny-squad"
        )
        self.tokenizer = qa_pipeline.tokenizer
        model_params = zoo_params[index]["model"]
        model_name = create_name(zoo_params[index])
        model_params["pipeline"] = qa_pipeline
        chkp_folder = "chkp"
        checkpoint = torch.load(os.path.join(chkp_folder, model_name, f'{model_name}.pt'))
        self.model = gptKILLER(**model_params)
        self.model.load_state_dict(checkpoint['model_state_dict'])
    
    def __call__(self, context, question):
        tokens = self.tokenizer(
            question,
            context, 
            return_tensors="pt"
        )

        with torch.no_grad():
            res = self.model(
                tokens.input_ids, 
                tokens.token_type_ids, 
                tokens.attention_mask, 
                torch.Tensor([0]).long(), 
                torch.Tensor([0]).long()
            )
        answer_start_index = res.start_logits.argmax()
        answer_end_index = res.end_logits.argmax()
        
        predict_answer_tokens = tokens.input_ids[0, answer_start_index : answer_end_index + 1]
        
        answer = self.tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)
        start_answer, end_answer = find_start_and_end(context, answer)
        answer = context[start_answer: end_answer + 1]

        return answer


def find_start_and_end(text, answer):
    if len(answer) == 0:
        return 0, -1
    i = 0
    j = 0
    additional_spaces = 0
    dash_count = 0
    skip = 0
    while i < len(answer):
        if text[j] == answer[i]:
            i += 1
        elif text[j] == " " and answer[i] == ",":
            skip += 1
        elif text[j] == answer[i + 1]:
            additional_spaces += 1
            i += 2
        elif text[j] == "–":
            dash_count += 2
            j += 1
        elif text[j] == answer[0]:
            i = 1
            additional_spaces = 0
        else:
            i = 0
            additional_spaces = 0
            dash_count = 0
            skip = 0
        j += 1
    return j - i + additional_spaces - skip - dash_count, j - 1


# def load_model(params, model_name, dir):
#     pass

# def create_pipeline(count_of_models=1):
#     best_params_for_1 = {}
#     best_params_for_2 = {}
#     if count_of_models == 2:
