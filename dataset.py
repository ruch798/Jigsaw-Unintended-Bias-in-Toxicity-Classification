import config
import torch
import transformers
from transformers import AutoTokenizer

class JigsawDataset:
    def __init__(self, comment_text, target, model_name):
        self.comment_text = comment_text
        self.target = target
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.max_len = config.MAX_LEN
        self.use_token_type_ids = "roberta" not in model_name.lower() and "distilbert" not in model_name.lower()

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, item):
        comment_text = str(self.comment_text[item])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=self.use_token_type_ids
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs.get("token_type_ids")

        sample = {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }

        if token_type_ids is not None:
            sample["token_type_ids"] = torch.tensor(token_type_ids, dtype=torch.long)

        return sample

class JigsawDatasetTest:
    def __init__(self, comment_text, tokenizer):
        self.comment_text = comment_text
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, item):
        comment_text = str(self.comment_text[item])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=config.MAX_LEN,
            padding='max_length',
            truncation=True,
        )
        
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long)
        }