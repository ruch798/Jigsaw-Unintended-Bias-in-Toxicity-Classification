import transformers

DEVICE = "cuda"
MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 3
NUM_SAMPLES = 50000
model1 = "bert-base-uncased"
model2 = "roberta-base"
model3 = "distilbert-base-uncased"
MODEL_DIR = "models/"
TRAINING_FILE = "data/train.csv"
TESTING_FILE = "data/test.csv"
IDENTITY_COLUMNS = ['male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish', 'muslim', 'black', 'white', 'psychiatric_or_mental_illness']
TOXICITY_COLUMN = 'target'
