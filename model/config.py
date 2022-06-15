# Model Configurations for Musical For Beginners
class MfbConfig:
    PRETRAINED = 'KOBERT'
    SAVE_MODEL_PATH = './finetuned_model/mfb_torch_model.pt'

    TEST_SIZE = 0.1

    LOG_INTERVAL = 8
    TRAIN_BATCH_SIZE = 8
    VALID_BATCH_SIZE = 4
    MAX_LEN = 128
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1
    LEARNING_RATE = 5e-06
    RANDOM_STATE = 42
    DR_RATE = 0.45
    EPOCHS = 20
    DEVICE = 'cpu'
    OUTPUT_LABEL_NUM = 8


# set collection names and label field name at DB where the annotated data at(we used MongoDB)
class LabelPool:
    TEXT_COL = 'review'
    LABEL_COL = 'label'

# set collection name where unseen data at
class MongoCollection:
    TEXT_COL = 'review'