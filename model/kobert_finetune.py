import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup

from metric_utils import LabelWiseMetrics

from config import MfbConfig, LabelPool

import pandas as pd
import re
import emoji
from soynlp.normalizer import repeat_normalize


from easy_dat_aug import EDA

emojis = list({y for x in emoji.UNICODE_EMOJI.values() for y in x.keys()})
emojis = ''.join(emojis)
pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣{emojis}]+')
url_pattern = re.compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

def clean(x):
    x = pattern.sub(' ', x)
    x = url_pattern.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x



_TEST_SIZE = 0.1

log_interval = 10
device = 'cpu'#'cuda' if torch.cuda.is_available() else 'cpu'

class KoBERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair, unseen=False):

        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([clean(i)]) for i in dataset[sent_idx]]

        if not unseen:
            self.labels = [np.int32(i) for i in dataset[label_idx]]
        else:
            self.labels = [np.zeros(MfbConfig.OUTPUT_LABEL_NUM) for i in range(len(dataset[sent_idx]))]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return (len(self.labels))



class KoBERTMfbModel(torch.nn.Module):
    def __init__(self,
                 base_model,
                 hidden_size = 768,
                 num_classes=MfbConfig.OUTPUT_LABEL_NUM,
                 dr_rate=None,
                 params=None):

        super(KoBERTMfbModel, self).__init__()

        self.layer1 = base_model
        self.layer2 = torch.nn.Linear(hidden_size, MfbConfig.OUTPUT_LABEL_NUM)
        self.dr_rate = dr_rate
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_len):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_len):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.layer1(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out2 = self.dropout(pooler)
        else:
            out2 = pooler
        out = self.layer2(out2)
        return out



def kobert_finetune(parameters, augmentation=False, rescaling=False):
    bertmodel = parameters['model']
    vocab = parameters['vocab']
    config = parameters['config']
    dataset = parameters['dataset']
    tokenizer = parameters['tokenizer']
    tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

    model = KoBERTMfbModel(bertmodel, num_classes=MfbConfig.OUTPUT_LABEL_NUM, dr_rate=MfbConfig.DR_RATE)
    model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=MfbConfig.LEARNING_RATE)



    train_set, test_set = train_test_split(dataset, test_size=_TEST_SIZE, random_state=MfbConfig.RANDOM_STATE, shuffle=True)

    if augmentation:
        tmp = list()
        for row in train_set.iloc():
            tmp.append({'review': row['review'], 'label': row['label']})
            aug = EDA(row['review'])
            for a in aug:
                tmp.append({'review': a, 'label': row['label']})

        train_set = pd.DataFrame(tmp)
        train_set = train_set.sample(frac=1)

    train_set.reset_index(inplace=True, drop=True)
    test_set.reset_index(inplace=True, drop=True)


    if rescaling:
        w = np.zeros(MfbConfig.OUTPUT_LABEL_NUM)
        for i in range(len(train_set)):
            w += train_set['label'][i]
        w = max(w) / w
        w = torch.Tensor(w)
        loss_fn = nn.MultiLabelSoftMarginLoss(weight=w.to(device))
    else:
        loss_fn = nn.MultiLabelSoftMarginLoss()


    data_train = KoBERTDataset(train_set, LabelPool.TEXT_COL, LabelPool.LABEL_COL, tok, MfbConfig.MAX_LEN, True, False)
    data_test = KoBERTDataset(test_set, LabelPool.TEXT_COL, LabelPool.LABEL_COL, tok, MfbConfig.MAX_LEN, True, False)

    train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=MfbConfig.TRAIN_BATCH_SIZE, num_workers=0)
    test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=MfbConfig.VALID_BATCH_SIZE, num_workers=0)

    t_total = len(train_dataloader) * MfbConfig.EPOCHS
    warmup_step = int(t_total * MfbConfig.WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)


    def train(n_epochs, train_dataloader, test_dataloader):
        accuracy_history = []
        loss_history = []
        recall_history = []
        precision_history = []
        f1_history = []
        for e in range(n_epochs):
            train_metric = LabelWiseMetrics()
            test_metric = LabelWiseMetrics()

            model.train()

            for e in range(n_epochs):
                train_metric = LabelWiseMetrics()
                test_metric = LabelWiseMetrics()

                model.train()
                train_loss = 0.0
                for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(train_dataloader)):
                    optimizer.zero_grad()
                    token_ids = token_ids.long().to(device)
                    segment_ids = segment_ids.long().to(device)
                    valid_length = valid_length
                    label = label.long().to(device)
                    out = model(token_ids, valid_length, segment_ids)
                    loss = loss_fn(out, label.float())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MfbConfig.MAX_GRAD_NORM)
                    optimizer.step()
                    scheduler.step()

                    if device == 'cuda':
                        out = out.cpu()
                        label = label.cpu()
                    pred = np.array(out.detach().numpy() > 0.5)
                    obs = label.detach().numpy()

                    train_metric.update(pred, obs)

                    train_acc = train_metric.get_accuracy()
                    train_loss += loss.data.cpu().numpy()

                avg_train_loss = train_loss / (batch_id + 1)
                print("epoch {} train avg loss {} acc {}".format(e + 1, avg_train_loss, train_metric.get_accuracy()))

                model.eval()
                test_loss = 0.0
                for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(test_dataloader)):
                    token_ids = token_ids.long().to(device)
                    segment_ids = segment_ids.long().to(device)
                    valid_length = valid_length
                    label = label.long().to(device)
                    out = model(token_ids, valid_length, segment_ids)
                    loss_val = loss_fn(out, label.float())

                    if device == 'cuda':
                        out = out.cpu()
                        label = label.cpu()

                    pred = np.array(out.detach().numpy() > 0.5)
                    obs = label.detach().numpy()

                    test_metric.update(pred, obs)

                    test_acc = test_metric.get_accuracy()
                    test_loss += loss_val.data.cpu().numpy()

                avg_test_loss = test_loss / (batch_id + 1)
                avg_test_acc = test_metric.get_accuracy()
                print("epoch {} loss {} test acc {}".format(e + 1, avg_test_loss, avg_test_acc))

                accuracy_history.append({'train': train_metric.get_accuracy(), 'test': test_metric.get_accuracy()})
                loss_history.append({'train': avg_train_loss, 'test': avg_test_loss})
                recall_history.append({'train': train_metric.get_recall(), 'test': test_metric.get_recall()})
                precision_history.append({'train': train_metric.get_precision(), 'test': test_metric.get_precision()})
                f1_history.append({'train': train_metric.get_f1(), 'test': test_metric.get_f1()})

    train(MfbConfig.EPOCHS, train_dataloader, test_dataloader)
    save_model(bertmodel, model, tok)


def save_model(base, trained, tok):
    torch.save({'base': base, 'model_state_dict': trained.state_dict(), 'tok': tok}, MfbConfig.SAVE_MODEL_PATH)

if __name__== '__main__':
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print('device: ', device)
    from mfb_fine_tune import FineTuningTrainer
    FineTuningTrainer('KOBERT', augmentation=True)

