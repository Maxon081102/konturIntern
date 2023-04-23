from torch.utils.data import Dataset

class ContractsDataset(Dataset):
    def __init__(self, dt, tokenizer):
        super(ContractsDataset).__init__()
        self.contracts = dt.text.to_list()
        self.labels = dt.label.to_list()
        self.answers = dt.extracted_part.to_list()
        self.tokenizer = tokenizer
        self.starts = []
        self.ends = []
        for i in range(len(self.labels)):
            start, end = find_answer(
                tokenizer, 
                self.labels[i],
                self.contracts[i],
                self.answers[i]["text"][0]
            )
            if start == -1:
                print(dt.unique_id.to_list()[i])
            self.starts.append(start)
            self.ends.append(end)
    
    def __getitem__(self, index):
        tokens = self.tokenizer(
            self.labels[index],
            self.contracts[index], 
            padding="max_length", 
            return_tensors="pt",
            max_length=1100
        )
        return tokens.input_ids[0], tokens.token_type_ids[0], tokens.attention_mask[0], \
            self.starts[index], self.ends[index]
    
    def __len__(self):
        return len(self.labels)

def find_answer(tokenizer, question, text, answer):
    if len(answer) == 0:
        return 0, 0
    text_tokens = tokenizer(question, text).input_ids
    answer_tokens = tokenizer(answer).input_ids[1:-1]
    i = 0
    j = 0
    while i < len(answer_tokens):
        if text_tokens[j] == answer_tokens[i]:
            i += 1
        elif text_tokens[j] == answer_tokens[0]:
            i = 1
        else:
            i = 0
        j += 1
        
        if j == len(text_tokens):
            print("skip")
            return -1, -1
    return j - i, j - 1
    