

_base_model_supported = ['KOBERT', 'KCBERT']
_hf_model_dir = {'KOBERT': 'skt/kobert-base-v1', 'KCBERT': 'beomi/kcbert-base'}


class PretrainedLoader:
    @staticmethod
    def load(model_name):
        if model_name not in _base_model_supported:
            print("Not Supported base model. (Supported Parameter: ", _base_model_supported, ")")
            raise Exception()

        model, vocab = load_pretrained_model(model_name)
        tokenizer = load_pretrained_tokenizer(model_name)
        config = load_pretrained_config(model_name)
        print(tokenizer)
        return {'model': model, 'vocab': vocab, 'tokenizer': tokenizer, 'config': config}


def load_pretrained_model(base_model_name):
    model = None

    if base_model_name == _base_model_supported[0]:
        #print('import BertModel from huggingface transformers')
        from kobert import get_pytorch_kobert_model
        print('Now Loading model ',base_model_name)
        model, vocab = get_pytorch_kobert_model()
        print('Model successfully loaded')
    elif base_model_name == _base_model_supported[1]:
        #print('import BertForSequenceClassification from huggingface transformers')
        from transformers import BertForSequenceClassification
        print('Now Loading model ', base_model_name)
        model = BertForSequenceClassification.from_pretrained(_hf_model_dir[base_model_name],
                                                              problem_type='multi_label_classification', num_labels=8)

        vocab = None
        print('Model successfully loaded')

    return model, vocab

def load_pretrained_tokenizer(base_model_name):
    from transformers import AutoTokenizer
    tokenizer = None

    if base_model_name == _base_model_supported[0]:
        from kobert import get_tokenizer
        print('Now Loading tokenizer ', base_model_name)
        tokenizer = get_tokenizer()
        print('Tokenizer successfully loaded')
    elif base_model_name == _base_model_supported[1]:
        from transformers import AutoTokenizer
        print('Now Loading tokenizer ', base_model_name)
        tokenizer = AutoTokenizer.from_pretrained(_hf_model_dir[base_model_name])
        print('Tokenizer successfully loaded')

    return tokenizer


def load_pretrained_config(base_model_name):
    from transformers import BertConfig
    if base_model_name == 'KOBERT':
        return None
    else:
        return BertConfig.from_pretrained(_hf_model_dir[base_model_name])


if __name__ == '__main__':
    PretrainedLoader.load('KOBERT')
