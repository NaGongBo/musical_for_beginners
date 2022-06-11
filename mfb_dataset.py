from mongo_access import mongo_auth_access
import pandas as pd
from model_io_translation import convert_record_to_label

_LABEL_DB_NAME = 'labeling_pool'
_LABEL_COL_NAME = 'test_collection'


class MfbDataset:
    def __init__(self):
        self.df = None

        self.collection = None
        self.skip_count=0
        self.label_count = {'funny': 0, 'touching': 0, 'story': 0, 'immersion': 0,
                            'stage': 0, 'song': 0, 'dance': 0, 'acting': 0}

    def __len__(self):
        assert self.df is not None
        return len(self.df)

    def load(self, skip_meaningless=False, count=True ,augmentation=False):

        # self.df = pd.read_json('./data.json')
        # for index, row in self.df.iterrows():
        #     tmp = row['review']
        #     print(tmp , row['label'])
        #
        #
        # return

        print('loading data from server')
        mongo_client = mongo_auth_access() # authorized access for security. (call MongoClient())

        self.collection = mongo_client[_LABEL_DB_NAME][_LABEL_COL_NAME]

        cursor = self.collection.find({'labeled' : {'$type': 'object'}})
        tmp_lst = list()
        self.skip_count = 0
        for doc in cursor:
            text = doc['review']
            label = doc['labeled']

            if count:
                for k in self.label_count:
                    self.label_count[k] += (label[k] != 0)

            tmp = label.values()
            meaningless = 1 not in tmp

            if skip_meaningless and meaningless:
                self.skip_count += 1
                continue
            else:
                label_tmp = convert_record_to_label(label)
                tmp_lst.append({'review' : text, 'label' : label_tmp})


        self.df = pd.DataFrame(tmp_lst)

        mongo_client.close()

    def show(self):
        assert self.df is not None
        print('skipped ',self.skip_count,' meaningless records')
        print(self.df)
        print(self.label_count)


if __name__ == '__main__':
    dataset = MfbDataset()
    dataset.load(skip_meaningless=True)
    dataset.show()
    dataset.df.to_json('./data.json')