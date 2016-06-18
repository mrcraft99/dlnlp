import cPickle
import numpy as np
import sys, re
import pandas as pd


def build_sentences(train_file,test_file):
    """
    Loads MSRP datasets
    """
    #filename="c:/Users/MrCraft/Desktop/ARCI/msr_paraphrase_train.txt"
    f=open(train_file)
    f.readline()
    lines=f.readlines()
    labels_train=[]
    sentence_pairs_train=[]
    vocab={}
    max_length=0;
    for line in lines:
        line_split=line.split('\t')
        labels_train.append(line_split[0])
        sentence_pairs_train.append([clean_str(line_split[3]),clean_str(line_split[4])])
    for pair in sentence_pairs_train:
        sentence1=pair[0].split(' ')
        sentence2=pair[1].split(' ')
        max_length=max(max_length,max(len(sentence1),len(sentence2)))
        for word in sentence1:
            vocab[word]=1
        for word in sentence2:
            vocab[word]=1
    f.close()

    f=open(test_file)
    f.readline()
    lines=f.readlines()
    labels_test=[]
    sentence_pairs_test=[]
    for line in lines:
        line_split=line.split('\t')
        labels_test.append(line_split[0])
        sentence_pairs_test.append([clean_str(line_split[3]),clean_str(line_split[4])])
    for pair in sentence_pairs_test:
        sentence1=pair[0].split(' ')
        sentence2=pair[1].split(' ')
        max_length=max(max_length,max(len(sentence1),len(sentence2)))
        for word in sentence1:
            vocab[word]=1
        for word in sentence2:
            vocab[word]=1
    f.close()
    return sentence_pairs_train,sentence_pairs_test,labels_train,labels_test,vocab,max_length


def load_bin_vec(filename, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs

def add_unknown_words(word_vecs, vocab):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs :
            word_vecs[word] = np.random.uniform(-0.25, 0.25, 300).astype('float32')

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size + 1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map


def to_idx(sentence_pairs_train,sentence_pairs_test, word_idx_map, max_length):
    """
    Transforms sentences into a 2-d matrix.
    """
    sentence_pairs_idx_train, sentence_pairs_idx_test = [], []
    for sentence_pair in sentence_pairs_train:
        sentence_pair_idx = to_idx_padded(sentence_pair, word_idx_map, max_length)
        sentence_pairs_idx_train.append(sentence_pair_idx)
    for sentence_pair in sentence_pairs_test:
        sentence_pair_idx=to_idx_padded(sentence_pair,word_idx_map,max_length)
        sentence_pairs_idx_test.append(sentence_pair_idx)
    train = np.array(sentence_pairs_idx_train,dtype="int")
    test = np.array(sentence_pairs_idx_test,dtype="int")
    return train, test

def to_idx_padded(sentence_pair, word_idx_map, max_length):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    sentence_pair_idx=[]
    for sentence in sentence_pair:
        x = []
        words = sentence.split()
        for word in words:
            if word in word_idx_map:
                x.append(word_idx_map[word])
        while len(x) < max_length:
            x.append(0)
        sentence_pair_idx.append(x)
    return sentence_pair_idx


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


if __name__ == "__main__":
    w2v_file = 'GoogleNews-vectors-negative300.bin'
    train_file='msr_paraphrase_train.txt'
    test_file='msr_paraphrase_test.txt'
    print "loading data..."
    sentence_pairs_train,sentence_pairs_test,labels_train,labels_test,vocab,max_length=build_sentences(train_file,test_file)
    print('numbers of sentence pairs in training set:'+str(len(sentence_pairs_train)))
    print('numbers of sentence pairs in test set:'+str(len(sentence_pairs_test)))
    print('max length:'+str(max_length))
    word_vecs=load_bin_vec(w2v_file,vocab)
    add_unknown_words(word_vecs, vocab)
    W, word_idx_map = get_W(word_vecs)
    sentence_pairs_idx_train,sentence_pairs_idx_test=to_idx(sentence_pairs_train,sentence_pairs_test,word_idx_map,max_length)
    cPickle.dump([sentence_pairs_idx_train,sentence_pairs_idx_test,labels_train,labels_test,W], open("data", "wb"))
    print('finished')



