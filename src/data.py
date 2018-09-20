import tensorflow as tf
from tf.keras.preprocessing.text import Tokenizer
from CONSTANTS import *

def read(data_file):

    data = open(data_file).read().strip()
    english_l = []
    hindi_l = []
    data_lists = data.split('\t')

    for i in data_lists:
        line = i.split('\t')
        line0 = 'S_ '+line[0]+ ' E_'
        line1 = 'S_ '+line[1]+' _E'
        english_l.append(line0)
        hindi_l.append(line1)

    return english_l,hindi_l

def indexize(documents):

    tokenizer = Tokenizer(oov_token='UNK',num_words=10000)
    tokenizer.fit_on_texts(documents)
    docs = tokenizer.texts_to_sequences(documents)

    return tokenizer,docs

def max_length(docs):

    f = lambda x : len(x)
    return max(list(map(f,docs)))

def one_hot(tokenizer,documents):

    """
    In case using categorical instead of sparse_categorical
    """

    return tokenizer.texts_to_matrix(documents,mode='binary')

def create_tensors(max_lengths,english_l,hindi_l):

    english_l = tf.keras.preprocessing.sequence.pad_sequences(english_l,
                                                              maxlen = max_length[0],
                                                              padding='post')
    hindi_l = tf.keras.preprocessing.sequence.pad_sequences(hindi_l,
                                                            maxlen=max_length[1],
                                                            padding='post')
    
    return english_l,hindi_l

class Dataset:

    def __init__(self,BUFFERSIZE,BATCH_SIZE,x,y):

        self.buffersize = BUFFERSIZE
        self.batch_size = BATCH_SIZE
        self.x = x
        self.y = y
        self.dataset = create_dataset()

    def create_dataset(self):

        self.dataset = tf.data.Dataset.from_tensor_slices((self.x,self.y))
        self.dataset = self.dataset.batch(self.batch_size)