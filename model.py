import config
import transformers
from transformers import AutoConfig, AutoModel
import torch
import torch.nn as nn


class JigsawModel(nn.Module):
    def __init__(self, model_name):
        super(JigsawModel, self).__init__()
        self.config = transformers.AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.model = transformers.AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(self.config.hidden_size, 1)

    def forward(self, ids, mask, token_type_ids=None):
        is_distilbert = "distilbert" in self.config.architectures[0].lower()
        if token_type_ids is not None:
            outputs = self.model(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)
        else:
            outputs = self.model(ids, attention_mask=mask, return_dict=False)
        
        if is_distilbert:
            o2 = outputs[0][:, 0, :]
        else:
            _, o2 = outputs

        bo = self.drop(o2)
        output = self.out(bo)
        
        return output
