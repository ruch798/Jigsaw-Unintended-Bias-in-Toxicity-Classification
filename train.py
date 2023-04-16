import config
import dataset
import engine
import final_metric
import torch
import pandas as pd
import numpy as np
import os
import torch.nn as nn
import transformers

from model import JigsawModel
from sklearn import model_selection
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import warnings
warnings.simplefilter('ignore')

def run(model_name):
    dfx = pd.read_csv(config.TRAINING_FILE).fillna("none")
    dfx = dfx.sample(frac=1, random_state=7).reset_index(drop=True)
    dfx = dfx.head(config.NUM_SAMPLES)
    dfx.target = dfx.target.apply(lambda x: 1 if x > 0.5 else 0)

    df_train, df_valid = model_selection.train_test_split(
        dfx, test_size=0.1, random_state=42, stratify=dfx.target.values
    )

    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.JigsawDataset(
        comment_text=df_train.comment_text.values, target=df_train.target.values, model_name= model_name 
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, num_workers=4
    )

    valid_dataset = dataset.JigsawDataset(
        comment_text=df_valid.comment_text.values, target=df_valid.target.values, model_name= model_name
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.VALID_BATCH_SIZE, num_workers=1
    )

    device = torch.device(config.DEVICE)
    model = JigsawModel(model_name)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = torch.optim.AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    model = nn.DataParallel(model)

    print(f"\n{'--'*5} MODEL: {model_name} {'--'*5}\n")
    best_roc_auc = 0
    valid_outputs = np.zeros(len(df_valid))
    for epoch in range(config.EPOCHS):
        print(f"\n{'--'*5} EPOCH: {epoch + 1} {'--'*5}\n")
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)
        targets = np.array(targets) >= 0.5
        roc_auc = metrics.roc_auc_score(targets, outputs)
        print(f"AUC Score = {roc_auc}")
        if roc_auc > best_roc_auc:
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, model_name + ".bin") )
            valid_outputs = np.array(outputs)
            best_roc_auc = roc_auc

    metric_value = final_metric.get_value(df_valid, valid_outputs, model_name)
    return metric_value

if __name__ == "__main__":
    model_names = [config.model1, config.model2, config.model3] # list of machine learning models  
    final_metrics = [] # list to store the final evaluation metric for each model

    # Iterate through the list of models
    for model_name in model_names:
        # Train and evaluate the model using the run() function
        metric = run(model_name)
        final_metrics.append(metric) # Store the final evaluation metric for the current model
        
    # Find the index of the model 
    # with the highest performance
    best_model_idx = final_metrics.index(max(final_metrics))

    # Print the name or information about the best performing model
    best_model = model_names[best_model_idx]
    print(f"\nThe best performing model is {best_model} with a final metric of {final_metrics[best_model_idx]}")

