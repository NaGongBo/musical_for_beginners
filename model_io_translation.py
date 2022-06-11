import copy
_base_label = {
    'funny': None,
    'touching': None,
    'story': None,
    'immersion': None,
    'stage': None,
    'song': None,
    'dance': None,
    'acting': None
}

_label_order = ['funny', 'touching', 'story', 'immersion', 'stage', 'song', 'dance', 'acting']


# convert dictionary type record from db to array like output
def convert_record_to_label(record):
    tmp_label = list()
    for i in range(len(_label_order)):
        val_tmp = record[_label_order[i]]
        if val_tmp == -1:
            val_tmp = 0
        tmp_label.append(val_tmp)

    return tmp_label


# convert model_output to dictionary type with keys
def convert_label_to_record(label):
    tmp_record = copy.deepcopy(_base_label)

    for i in range(len(_label_order)):
        tmp_record[_label_order[i]] = label[i]

    return tmp_record