from .pretrained_loader import PretrainedLoader
import torch
from ..mfb_dataset import MfbDataset
from .config import MfbConfig


"""
parameters to fine-tuning routine will be passed by dictionary instance
e.g. {'model': , 'tokenizer': , 'config': , 'dataset': }
"""


class FineTuningTrainer:
    def __init__(self, base_model_name, augmentation=False):
        arg = PretrainedLoader.load(base_model_name)
        data = MfbDataset()
        data.load(skip_meaningless=True)
        arg.update({'dataset': data.df})

        if base_model_name == 'KOBERT':
            from kobert_finetune import kobert_finetune
            kobert_finetune(arg, augmentation=augmentation)
        elif base_model_name =='KCBERT':
            from kcbert_finetune import kcbert_finetune
            kcbert_finetune(arg)


if __name__ == '__main__':

    FineTuningTrainer(MfbConfig.PRETRAINED)
