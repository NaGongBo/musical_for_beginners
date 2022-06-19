import numpy as np
from sklearn.metrics import confusion_matrix
from .config import MfbConfig

####################################
# update the confusion matrix from the prediction and label.
# then calculate all metrics per label.
####################################

TP = (1, 1)
FN = (1, 0)
FP = (0, 1)
TN = (0, 0)

class LabelWiseMetrics:

    def __init__(self):

        self.conf_mat_list = list()
        for i in range(MfbConfig.OUTPUT_LABEL_NUM):
            self.conf_mat_list.append(np.zeros((2,2)))

        self.conf_mat_list = list()
        for i in range(MfbConfig.OUTPUT_LABEL_NUM):
            self.conf_mat_list.append(np.zeros((2,2)))
        self.acc = np.zeros(MfbConfig.OUTPUT_LABEL_NUM)
        self.recall = np.zeros(MfbConfig.OUTPUT_LABEL_NUM)
        self.precision = np.zeros(MfbConfig.OUTPUT_LABEL_NUM)
        self.f1 = np.zeros(MfbConfig.OUTPUT_LABEL_NUM)



    def update(self, prediction, truth):
        y_pred = prediction.T
        y_true =  truth.T
        for i in range(MfbConfig.OUTPUT_LABEL_NUM):
            self.conf_mat_list[i] += confusion_matrix(y_true[i], y_pred[i], labels=[0,1])

        for i in range(MfbConfig.OUTPUT_LABEL_NUM):
            conf = self.conf_mat_list[i]

            rc_avail = (conf[TP] + conf[FN] != 0)
            pr_avail = (conf[TP] + conf[FP] != 0)
            acc_avail = (conf.sum() != 0)
            f1_avail = (2*conf[TP] + conf[FP] + conf[FN] != 0)

            self.acc[i] = (conf[TP] + conf[TN]) / (conf.sum()) if acc_avail else None
            self.recall[i] = conf[TP] / (conf[TP] + conf[FN]) if rc_avail else None
            self.precision[i] = conf[TP] / (conf[TP] + conf[FP]) if pr_avail else None
            self.f1[i] = (2*conf[TP]) / (2*conf[TP] + conf[FP] + conf[FN]) if f1_avail else None


    def get_accuracy(self):
        return self.acc

    def get_recall(self):
        return self.recall

    def get_precision(self):
        return self.precision

    def get_f1(self):
        return self.f1

    def get_metrics(self):
        return{
            'accuracy': self.get_accuracy(),
            'recall': self.get_recall(),
            'precision': self.get_precision(),
            'f-1': self.get_f1()
        }
