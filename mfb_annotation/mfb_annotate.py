# simple gui workspace for annotation on review data from mongodb
# implemented to create labeled dataset for Musical for Beginners
import sys
from copy import deepcopy

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from mongo_access import mongo_auth_access

import re

_CLIENT_AUTH = {"_id" : 'dongi_admin', "_passwd": "djWjekehdrl_capstone1"}

_BASE_FEATURES = {
    'funny': '유쾌',
    'touching': '감동',
    'story': '스토리',
    'immersion': '몰입도',
    'stage': '무대',
    'song': '노래',
    'dance': '춤',
    'acting': '연기'
}
_FEATURE_REPRESENT = {'선택': None, '긍정적': 1, '중립': 0, '부정적': -1}

_base_record = {
    'funny': None,
    'touching': None,
    'story': None,
    'immersion': None,
    'stage': None,
    'song': None,
    'dance': None,
    'acting': None
}


_RAW_DB_NAME = 'JJINMAK'
_RAW_COLL_NAME = 'review'
_RAW_CONTENT_FIELD_NAME = 'review'
_LABEL_DB_NAME = 'labeling_pool'
_LABEL_COL_NAME = 'test_collection'


class MongoPusher:
    def __init__(self):
        self.review = None
        self.review_id = None
        self.label_data = None

        self.db_connection = mongo_auth_access()
        self.init_new()

    def init_new(self):
        self.review = None
        self.review_id = None
        self.label_data = None

        _tmp = None
        pat = re.compile(r'웃', re.I)
        while True:
            _tmp_cursor = self.db_connection[_RAW_DB_NAME][_RAW_COLL_NAME].aggregate([{'$match': {'review': {'$regex':pat}}},{'$sample': {'size': 1}}])
            _tmp = _tmp_cursor.next()
            break
            #if _tmp['label'] == 0:
            #    self.db_connection[_RAW_DB_NAME][_RAW_COLL_NAME].find_one_and_update(
            #        {'_id': _tmp.get('_id')},
            #        {'$set': {'labeled' : -1}}
            #    )
            #    break
        self.review = _tmp.get(_RAW_CONTENT_FIELD_NAME)
        self.review_id = _tmp.get('_id')

    def refresh(self):
        del self.review
        del self.review_id
        del self.label_data
        self.init_new()

    def push_db(self):
        #이건 처음이랑 구조가 달라서 비효율적이어도 어쩔 수 없어용
        self.db_connection[_LABEL_DB_NAME][_LABEL_COL_NAME].find_one_and_update(
            {'_id': self.review_id},
            {'$set': {'labeled': self.label_data}}
        )

    def get_text(self):
        return self.review

    def set_data(self, new_record):
        self.label_data = new_record

    def terminate(self):
        self.db_connection[_RAW_DB_NAME][_RAW_COLL_NAME].find_one_and_update(
            {'_id': self.review_id},
            {'$set': {'labeled': 0}}
        )
        self.db_connection.close()



class MfbAnnotationGUI:

    def __init__(self):
        self.annotator = MongoPusher()
        self.app = QApplication([])

        self.review_content = QTextEdit()
        self.review_content.setReadOnly(True)
        self.review_content.setFixedSize(1600, 450)
        self.review_content.setText(self.annotator.get_text())
        self.cb_lst = list()
        self.cb_title = list()
        for k in _BASE_FEATURES:
            self.cb_title.append(k)
            tmp = QComboBox()
            tmp.setFixedSize(50,30)

            tmp.addItems(_FEATURE_REPRESENT.keys())
            self.cb_lst.append(tmp)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.review_content)
        self.window = QWidget()
        self.window.setFixedSize(1600, 900)
        self.window.setLayout(self.layout)

        for i in range(len(self.cb_title)):
            tmp = QLabel(_BASE_FEATURES[self.cb_title[i]])
            tmp_layout = QHBoxLayout()
            tmp_layout.setAlignment(Qt.AlignLeft)
            tmp_layout.addWidget(self.cb_lst[i])
            tmp_layout.addWidget(tmp)
            self.layout.addLayout(tmp_layout)

        self.button_layout = QHBoxLayout()
        next_but = QPushButton('계속')
        next_but.setFixedSize(50,30)
        next_but.clicked.connect(self.next_button_pushed)
        quit_but = QPushButton('끝내기')
        quit_but.setFixedSize(50,30)
        quit_but.clicked.connect(self.terminate)
        self.button_layout.addWidget(next_but,0)
        self.button_layout.addWidget(quit_but,1)
        self.button_layout.setAlignment(Qt.AlignLeft)
        self.layout.addLayout(self.button_layout)

        self.app.aboutToQuit.connect(self.terminate)
        self.window.show()

        self.msg_box = QMessageBox()
        self.msg_box.setText('5252 안채운게 있다구 휴먼')
        self.msg_box.addButton(QPushButton('거참 귀찮게 하네'), QMessageBox.YesRole)


    def exec(self):
        self.app.exec()

    def next_button_pushed(self):
        record = deepcopy(_base_record)
        for i in range(len(self.cb_title)):
            val = _FEATURE_REPRESENT[self.cb_lst[i].currentText()]
            if val is None:
                self.msg_box.exec()
                return
            else:
                record[self.cb_title[i]] = val

        for cb in self.cb_lst:
            cb.setCurrentIndex(0)
        self.annotator.set_data(record)
        self.annotator.push_db()
        self.annotator.refresh()

        self.review_content.setText(self.annotator.get_text())



    def terminate(self):
        self.annotator.terminate()
        sys.exit(0)


if __name__ == '__main__':
    MfbAnnotationGUI().exec()
