import config
from model import JigsawModel
import transformers
import torch
import streamlit as st

from transformers import AutoTokenizer, AutoModel

def sentence_prediction(sentence):
    model_name = 'distilbert-base-uncased'
    model = JigsawModel(model_name)
    model.load_state_dict(torch.load("models\distilbert-base-uncased.bin"), strict=False)
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    max_len = config.MAX_LEN
    comment_text = str(sentence)
    comment_text = " ".join(comment_text.split())

    inputs = tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length'
        )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)

    output = model(ids, mask)
    output = torch.sigmoid(output).cpu().detach().numpy()
    return output[0][0]

def main():
    st.title("Text Classification")
    sentence = st.text_input("Enter text:")
    if st.button("Classify"):
        prediction = sentence_prediction(sentence)
        if prediction > 0.5:
            st.write(f'This text is likely toxic')
        else:
            st.write(f'This text is likely not toxic')

if __name__ == "__main__":
    main()

