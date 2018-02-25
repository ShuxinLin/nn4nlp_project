from preprocessor import Preprocessor

data_path = "../dataset/CoNLL-2003/"
train_file = "eng.train"
val_file = "eng.testa"
test_file = "eng.testb"

def prepocess(file):
    preprocessor = Preprocessor(data_path, file)
    preprocessor.read_file()
    preprocessor.preprocess()
    preprocessor.index_preprocess()
    X, Y = preprocessor.minibatch()
    return X, Y
