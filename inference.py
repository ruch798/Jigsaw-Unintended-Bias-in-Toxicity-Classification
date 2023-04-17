import config
import dataset
from model import JigsawModel
import pandas as pd
import torch
import transformers
from tqdm import tqdm

def infer():
    test_df = pd.read_csv(config.TESTING_FILE).fillna("none")

    model_name = 'distilbert-base-uncased'
    model = JigsawModel(model_name).to(config.DEVICE)
    model.load_state_dict(torch.load("models\distilbert-base-uncased.bin"), strict=False)
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, do_lower_case=True)

    test_dataset = dataset.JigsawDatasetTest(
            comment_text=test_df.comment_text.values,
            tokenizer=tokenizer
    )

    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        drop_last=False,
        num_workers=2,
        shuffle=False
    )

    with torch.no_grad():
        fin_outputs = []
        for bi, d in tqdm(enumerate(test_data_loader), total = len(test_data_loader)):
            ids = d["ids"]
            mask = d["mask"]
            
            ids = ids.to(config.DEVICE, dtype=torch.long)
            mask = mask.to(config.DEVICE, dtype=torch.long)

            output = model(
                ids=ids,
                mask=mask
            )

            output = torch.sigmoid(output).cpu().detach().numpy()
            fin_outputs.extend(output.flatten().tolist())

    test_df['prediction'] = fin_outputs
    test_df.to_csv("data\predictions.csv")

if __name__ == "__main__":
    infer()