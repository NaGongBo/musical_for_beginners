import torch
from .config import MfbConfig, LabelPool, MongoCollection
import numpy as np
from ..model_io_translation import convert_label_to_record
import pandas as pd
from tqdm import tqdm

from .kobert_finetune import KoBERTDataset

device = 'cpu'


def make_prediction(model_output):
    return np.array(model_output.detach().cpu().numpy() > 0.5).astype(dtype=int)


class MusicalForBeginners:

    def __init__(self):

        print("Initializing Model")

        if MfbConfig.PRETRAINED == 'KOBERT':
            from .kobert_finetune import KoBERTMfbModel
            tmp = torch.load(MfbConfig.SAVE_MODEL_PATH)
            bert_model = tmp['base']
            tok = tmp['tok']
            model = KoBERTMfbModel(bert_model, num_classes=MfbConfig.OUTPUT_LABEL_NUM)
            model.load_state_dict(tmp['model_state_dict'])
        else:
            raise NotImplementedError

        self.model = model
        self.model.eval()
        self.model.to(device)
        self.tok = tok

        print("Successfully Initialized")

    def make_prediction_to_unseen_df(self, df):
        self.model.eval()
        df.reset_index(drop=True, inplace=True)
        data = KoBERTDataset(df, MongoCollection.TEXT_COL, None, self.tok, MfbConfig.MAX_LEN, True, False, unseen=True)
        data_loader = torch.utils.data.DataLoader(data, batch_size=1, num_workers=0)

        prediction_list = list()
        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(data_loader):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            out = self.model(token_ids, valid_length, segment_ids)
            prediction = make_prediction(out)
            prediction_list.append(prediction)


        total_prediction = prediction_list[0]
        for i in range(1, len(prediction_list)):
            total_prediction = np.append(total_prediction, prediction_list[i], axis=0)

        output = list()
        for row in total_prediction:
            output.append(convert_label_to_record(row.tolist()))

        output = pd.Series(output)
        df_tmp = df.assign(label=output)

        return df_tmp


    def make_prediction_to_usneen_string(self, input_str):
        df = pd.DataFrame([{'review': input_str}])
        return self.make_prediction_to_unseen_df(df)

    def validate_from_dataset(self):
        self.model.eval()
        from ..mfb_dataset import MfbDataset
        from .metric_utils import LabelWiseMetrics

        data = MfbDataset()
        data.load()
        data_test = KoBERTDataset(data.df, LabelPool.TEXT_COL, LabelPool.LABEL_COL, self.tok, MfbConfig.MAX_LEN, True, False)
        data_loader = torch.utils.data.DataLoader(data_test, batch_size=MfbConfig.VALID_BATCH_SIZE, num_workers=0)

        calc_metrics = LabelWiseMetrics()

        for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(tqdm(data_loader)):
            token_ids = token_ids.long().to(device)
            segment_ids = segment_ids.long().to(device)
            valid_length = valid_length
            label = label.long().to(device)
            out = self.model(token_ids, valid_length, segment_ids)
            out = out.cpu()
            prediction = make_prediction(out)
            true_val = make_prediction(label)

            for i in range(len(prediction)):
                print(prediction[i], true_val[i])

            calc_metrics.update(prediction, true_val)
            if batch_id % 50 == 0:
                print('\n',calc_metrics.get_accuracy())



        m = calc_metrics.get_metrics()
        for key in m:
            print(key, ': ', m[key])


if __name__ == '__main__':
    classifier = MusicalForBeginners()

    #from mfb_dataset import MfbDataset

    # data = MfbDataset()
    # data.load(skip_meaningless=True)
    # df = data.df[10:15]
    # df = classifier.make_prediction_to_unseen_df(df)


    a = None
    while a != 'exit':
        a = input()
        pred = classifier.make_prediction_to_usneen_string(a)
        print(pred['label'][0])