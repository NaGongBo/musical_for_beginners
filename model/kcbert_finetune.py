import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import TrainingArguments, Trainer
import numpy as np
#from sklearn import metrics
#from tqdm import tqdm

from .config import MfbConfig, LabelPool


import os




def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)


class KcBERTMfbDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):

        self.text = dataframe.review
        self.tokenizer = tokenizer
        self.targets = dataframe[LabelPool.LABEL_COL].to_list()
        self.max_len = max_len

    def __getitem__(self, index):

        review_text = str(self.text[index])
        review_text = " ".join(review_text.split())

        inputs = self.tokenizer(
            review_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True,
        )

        input_ids = inputs['input_ids']
        mask = inputs["attention_mask"]
        token_type_ids = inputs['token_type_ids']

        return{
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'labels': torch.tensor(self.targets[index], dtype=torch.long).float()
        }

    def __len__(self):
        return len(self.text)


class KcBERTMfbModel(torch.nn.Module):
    def __init__(self, base_model):
        super(KcBERTMfbModel, self).__init__()
        self.layer1 = base_model

    def forward(self, input_ids, attention_mask, token_type_ids, labels):

        output = self.layer1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        return output






class Arg:
    random_seed: int = 42  # Random Seed
    pretrained_model: str = 'beomi/kcbert-large'  # Transformers PLM name
    pretrained_tokenizer: str = ''  # Optional, Transformers Tokenizer Name. Overrides `pretrained_model`
    auto_batch_size: str = 'power'  # Let PyTorch Lightening find the best batch size
    batch_size: int = 0  # Optional, Train/Eval Batch Size. Overrides `auto_batch_size`
    lr: float = 5e-6  # Starting Learning Rate
    epochs: int = 20  # Max Epochs
    max_length: int = 150  # Max Length input size
    report_cycle: int = 100  # Report (Train Metrics) Cycle
    train_data_path: str = "nsmc/ratings_train.txt"  # Train Dataset file
    val_data_path: str = "nsmc/ratings_test.txt"  # Validation Dataset file
    cpu_workers: int = os.cpu_count()  # Multi cpu workers
    test_mode: bool = False  # Test Mode enables `fast_dev_run`
    optimizer: str = 'AdamW'  # AdamW vs AdamP
    lr_scheduler: str = 'exp'  # ExponentialLR vs CosineAnnealingWarmRestarts
    fp16: bool = False  # Enable train on FP16
    tpu_cores: int = 0  # Enable TPU with 1 core or 8 cores

args = Arg()



def main():
    print("Using PyTorch Ver", torch.__version__)
    print("Fix Seed:", args.random_seed)

    print(":: Start Training ::")
    trainer = Trainer(
        max_epochs=MfbConfig.EPOCHS,
        fast_dev_run=args.test_mode,
        num_sanity_val_steps=None if args.test_mode else 0,
        auto_scale_batch_size=args.auto_batch_size if args.auto_batch_size and not args.batch_size else False,
        # For GPU Setup
        deterministic= False,#torch.cuda.is_available(),
        gpus=-1 if torch.cuda.is_available() else None,
        precision=16 if args.fp16 else 32,
        device=MfbConfig.DEVICE,
        # For TPU Setup
        # tpu_cores=args.tpu_cores if args.tpu_cores else None,
    )


def compute_metrics(eval_pred):
    from datasets import load_metric

    metric = load_metric("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def kcbert_finetune(parameters):
    base_model = parameters['model']
    tokenizer = parameters['tokenizer']
    config = parameters['config']
    dataset = parameters['dataset']

    train_set, test_set = train_test_split(dataset, test_size=MfbConfig.TEST_SIZE, random_state=MfbConfig.RANDOM_STATE)
    train_set.reset_index(inplace=True)
    test_set.reset_index(inplace=True)

    train_data = KcBERTMfbDataset(train_set, tokenizer, MfbConfig.MAX_LEN)
    test_data = KcBERTMfbDataset(train_set, tokenizer, MfbConfig.MAX_LEN)

    train_params = {'batch_size': MfbConfig.TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': MfbConfig.VALID_BATCH_SIZE,
                   'shuffle': True,
                   'num_workers': 0
                   }



    model = base_model# KcBERTMfbModel(base_model)
    model.to(MfbConfig.DEVICE)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=MfbConfig.LEARNING_RATE)

    training_args = TrainingArguments(
        output_dir='./finetuned_model/kcbert_fine_tuned',
        per_device_train_batch_size=4,  # hyperparmeter
        per_device_eval_batch_size=1,  # hyperparmeter
        learning_rate=1e-5,  # hyperparmeter
        num_train_epochs=1,
        load_best_model_at_end=True,
        save_total_limit=2,
        evaluation_strategy="epoch",
        save_strategy='epoch',
        no_cuda=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()


def save_model(base, trained):
    torch.save({'base': base, 'model_state_dict': trained.state_dict()}, './finetuned_model/kcbert_based.pt')
