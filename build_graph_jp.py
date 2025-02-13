import os
import random

import numpy as np
import pickle as pkl
import networkx as nx
import scipy
import scipy.sparse as sp
from scipy.spatial.distance import cosine
import fasttext
import fasttext.util

import MeCab
from utils import loadWord2Vec, clean_str
from math import log
from sklearn import svm
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
from scipy.spatial.distance import cosine

if len(sys.argv) != 3:
    sys.exit("Use: python build_graph.py <dataset> <weight_mode>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'japanese', 'jp_full_label']

# fasttext download model
fasttext.util.download_model('en', if_exists='ignore')  # korean model
ft = fasttext.load_model('cc.en.300.bin')


def cos_similarity(word1, word2, fasttext_model):
    word1_emb = np.mean([fasttext_model[word1]], axis=0)
    word2_emb = np.mean([fasttext_model[word2]], axis=0)
    distance = scipy.spatial.distance.cosine(word1_emb, word2_emb)
    if distance >= 0:
        return 1 - distance
    else:
        return 0


def doc_word_similarity(word, doc, fasttext_model):
    word_emb = np.mean([fasttext_model[word]], axis=0)
    doc_emb = np.mean([fasttext_model[w] for w in doc], axis=0)
    distance = scipy.spatial.distance.cosine(word_emb, doc_emb)
    if distance >= 0:
        return 1 - distance
    else:
        return 0


# build corpus
dataset = sys.argv[1]
weight_mode = sys.argv[2]  # ['pmi', 'cos']
# dataset = 'japanese'

full_label = [
    '文学、本',
    '映画',
    '芸術、デザイン',
    'パフォーマンス、ショー',
    '音楽',
    'ドラマ',
    'スター、有名人',
    'アニメ',
    'テレビ、ラジオ放送',
    '日常',
    '子育て・結婚',
    'ペット',
    '良い記事、画像',
    'ファッション、美容',
    'インテリア、DIY',
    '料理レシピ',
    '製品レビュー',
    '園芸、栽培',
    'ゲーム',
    'スポーツ',
    '写真',
    '車',
    '趣味',
    '国内旅行',
    '海外旅行',
    'レストラン',
    'IT、コンピュータ',
    '社会、政治',
    '健康、医学',
    'ビジネス、経済',
    '言語、外国語',
    '教育、学術'
]

# if dataset not in datasets:
#     sys.exit("wrong dataset name")

# Read Word Vectors
# word_vector_file = 'data/glove.6B/glove.6B.300d.txt'
# word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
# _, embd, word_vector_map = loadWord2Vec(word_vector_file)
# word_embeddings_dim = len(embd[0])

word_embeddings_dim = 300
word_vector_map = {}  # maybe I can replace this with fasttext?

# shulffing
doc_name_list = []  # all documents title name
doc_train_list = []  # train documents title name
doc_test_list = []  # test documents title title
doc_val_list = []  # val documents title title

doc_content_list = []
train_content_list = []
val_content_list = []
test_content_list = []

both_path = 'data/japanese/'

ftrain = open(both_path + 'train.source', 'r', encoding='utf-8')
fval = open(both_path + 'val.source', 'r', encoding='utf-8')
ftest = open(both_path + 'test.source', 'r', encoding='utf-8')

lines = ftrain.readlines()
source_train_idx = []
source_val_idx = []
source_test_idx = []
for i, line in enumerate(lines):
    line = line.strip()
    title, content = line.split('\t')
    if len(content) > 0:
        source_train_idx.append(i)
        doc_name_list.append(title)
        doc_train_list.append(title)
        doc_content_list.append(content)
        train_content_list.append(content)
    # doc_name_list.append(line.strip())
    # doc_train_list.append(line.strip())
ftrain.close()

lines = fval.readlines()
for i, line in enumerate(lines):
    line = line.strip()
    title, content = line.split('\t')
    if len(content) > 0:
        source_val_idx.append(i)
        doc_name_list.append(title)
        doc_val_list.append(title)
        doc_content_list.append(content)
        val_content_list.append(content)
    # doc_name_list.append(line.strip())
    # doc_val_list.append(line.strip())
fval.close()

lines = ftest.readlines()
for i, line in enumerate(lines):
    line = line.strip()
    title, content = line.split('\t')
    if len(content) > 0:
        source_test_idx.append(i)
        doc_name_list.append(title)
        doc_test_list.append(title)
        doc_content_list.append(content)
        test_content_list.append(content)
    # doc_name_list.append(line.strip())
    # doc_test_list.append(line.strip())
ftest.close()

# doc_content_list = []
# train_content_list = []
# val_content_list = []
# test_content_list = []

# ftrain = open(content_path + 'train.source', 'r', encoding='utf-8')
# fval = open(content_path + 'val.source', 'r', encoding='utf-8')
# ftest = open(content_path + 'test.source', 'r', encoding='utf-8')

# lines = ftrain.readlines()
# counter = 0
# for line in lines:
#     counter += 1
#     doc_content_list.append(line.strip())
#     train_content_list.append(line.strip())
# ftrain.close()
#
# lines = fval.readlines()
# for line in lines:
#     doc_content_list.append(line.strip())
#     val_content_list.append(line.strip())
# fval.close()
#
# lines = ftest.readlines()
# for line in lines:
#     doc_content_list.append(line.strip())
#     test_content_list.append(line.strip())
# ftest.close()

# label list
# label_set = set()
train_label_list = []
val_label_list = []
test_label_list = []
doc_label_list = []

ftrain = open(both_path + 'train.target', 'r', encoding='utf-8')
fval = open(both_path + 'val.target', 'r', encoding='utf-8')
ftest = open(both_path + 'test.target', 'r', encoding='utf-8')

lines = ftrain.readlines()
for i, line in enumerate(lines):
    if i in source_train_idx:
        temp = line.split(">")
        # label_set.add(temp[0].strip())
        train_label_list.append(temp[0].strip())
        doc_label_list.append(temp[0].strip())
ftrain.close()

lines = fval.readlines()
for i, line in enumerate(lines):
    if i in source_val_idx:
        temp = line.split(">")
        # label_set.add(temp[0].strip())
        val_label_list.append(temp[0].strip())
        doc_label_list.append(temp[0].strip())
fval.close()

lines = ftest.readlines()
for i, line in enumerate(lines):
    if i in source_test_idx:
        temp = line.split(">")
        # label_set.add(temp[0].strip())
        test_label_list.append(temp[0].strip())
        doc_label_list.append(temp[0].strip())
ftest.close()

# if dataset == 'jp_full_label':
#     for label in full_label:
#         label_set.add(label.strip())
#
# label_list = list(label_set)

# partial labeled data
# train_ids = train_ids[:int(0.2 * len(train_ids))]
train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    train_ids.append(train_id)
print(train_ids)
random.shuffle(train_ids)

train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open('data/' + dataset + '.' + weight_mode + '.train.index', 'w', encoding='utf-8')
f.write(train_ids_str)
f.close()

val_ids = []
for val_name in doc_val_list:
    val_id = doc_name_list.index(val_name)
    val_ids.append(val_id)
print(val_ids)
random.shuffle(val_ids)

val_ids_str = '\n'.join(str(index) for index in val_ids)
f = open('data/' + dataset + '.' + weight_mode + '.val.index', 'w', encoding='utf-8')
f.write(val_ids_str)
f.close()

test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    test_ids.append(test_id)
print(test_ids)
random.shuffle(test_ids)

test_ids_str = '\n'.join(str(index) for index in test_ids)
f = open('data/' + dataset + '.' + weight_mode + '.test.index', 'w', encoding='utf-8')
f.write(test_ids_str)
f.close()

ids = train_ids + val_ids + test_ids
# print(ids)
print(len(ids))

shuffle_doc_name_list = []
shuffle_doc_words_list = []
shuffle_doc_label_list = []
for idx in ids:
    shuffle_doc_name_list.append(doc_name_list[int(idx)])
    if len(doc_content_list[int(idx)]) < 1:
        print("DEBUG doc_content_list")
        print(doc_content_list[int(idx)])
        print("title", str(doc_name_list[int(idx)]))
        exit()
    shuffle_doc_words_list.append(doc_content_list[int(idx)])
    shuffle_doc_label_list.append(doc_label_list[int(idx)])
shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
shuffle_doc_words_str = '\n'.join(shuffle_doc_words_list)
shuffle_doc_label_str = '\n'.join(shuffle_doc_label_list)

f = open('data/' + dataset + '.' + weight_mode + '_shuffle.txt', 'w', encoding='utf-8')
f.write(shuffle_doc_name_str)
f.close()

f = open('data/corpus/' + dataset + '.' + weight_mode + '_shuffle.txt', 'w', encoding='utf-8')
f.write(shuffle_doc_words_str)
f.close()

f = open('data/corpus/' + dataset + '.' + weight_mode + 'label_shuffle.txt', 'w', encoding='utf-8')
f.write(shuffle_doc_label_str)
f.close()

# build vocab
word_freq = {}
word_set = set()
mecab_tagger = MeCab.Tagger("-Owakati")
for doc_words in shuffle_doc_words_list:
    words = mecab_tagger.parse(doc_words).split()
    for word in words:
        word_set.add(word)
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

vocab = list(word_set)
vocab_size = len(vocab)

word_doc_list = {}

for i in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[i]
    words = mecab_tagger.parse(doc_words).split()
    appeared = set()
    for word in words:
        if word in appeared:
            continue
        if word in word_doc_list:
            doc_list = word_doc_list[word]
            doc_list.append(i)
            word_doc_list[word] = doc_list
        else:
            word_doc_list[word] = [i]
        appeared.add(word)

word_doc_freq = {}
for word, doc_list in word_doc_list.items():
    word_doc_freq[word] = len(doc_list)

word_id_map = {}
for i in range(vocab_size):
    word_id_map[vocab[i]] = i

vocab_str = '\n'.join(vocab)

f = open('data/corpus/' + dataset + '.' + weight_mode + '_vocab.txt', 'w', encoding='utf-8')
f.write(vocab_str)
f.close()

'''
Word definitions begin
'''
'''
definitions = []

for word in vocab:
    word = word.strip()
    synsets = wn.synsets(clean_str(word))
    word_defs = []
    for synset in synsets:
        syn_def = synset.definition()
        word_defs.append(syn_def)
    word_des = ' '.join(word_defs)
    if word_des == '':
        word_des = '<PAD>'
    definitions.append(word_des)

string = '\n'.join(definitions)


f = open('data/corpus/' + dataset + '_vocab_def.txt', 'w')
f.write(string)
f.close()

tfidf_vec = TfidfVectorizer(max_features=1000)
tfidf_matrix = tfidf_vec.fit_transform(definitions)
tfidf_matrix_array = tfidf_matrix.toarray()
print(tfidf_matrix_array[0], len(tfidf_matrix_array[0]))

word_vectors = []

for i in range(len(vocab)):
    word = vocab[i]
    vector = tfidf_matrix_array[i]
    str_vector = []
    for j in range(len(vector)):
        str_vector.append(str(vector[j]))
    temp = ' '.join(str_vector)
    word_vector = word + ' ' + temp
    word_vectors.append(word_vector)

string = '\n'.join(word_vectors)

f = open('data/corpus/' + dataset + '_word_vectors.txt', 'w')
f.write(string)
f.close()

word_vector_file = 'data/corpus/' + dataset + '_word_vectors.txt'
_, embd, word_vector_map = loadWord2Vec(word_vector_file)
word_embeddings_dim = len(embd[0])
'''

'''
Word definitions end
'''

# label list
# label_set = set()
# for doc_meta in shuffle_doc_name_list:
#     temp = doc_meta.split('\t')
#     label_set.add(temp[2])
# label_list = list(label_set)

# label_list_str = '\n'.join(label_list)
label_list_str = '\n'.join(full_label)
f = open('data/corpus/' + dataset + '.' + weight_mode + '_labels.txt', 'w', encoding='utf-8')
f.write(label_list_str)
f.close()

# x: feature vectors of training docs, no initial features
# slect 90% training set
train_size = len(train_ids)
# val_size = int(0.1 * train_size)
val_size = len(val_ids)
# real_train_size = train_size - val_size  # - int(0.5 * train_size)
real_train_size = train_size
# different training rates

# real_train_doc_names = shuffle_doc_name_list[:real_train_size]
# real_train_doc_names_str = '\n'.join(real_train_doc_names)
#
# f = open('data/' + dataset + '.real_train.name', 'w')
# f.write(real_train_doc_names_str)
# f.close()

row_x = []
col_x = []
data_x = []
for i in range(real_train_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = mecab_tagger.parse(doc_words).split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            # print(doc_vec)
            # print(np.array(word_vector))
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_x.append(i)
        col_x.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_x.append(doc_vec[j] / doc_len)  # doc_vec[j]/ doc_len

# x = sp.csr_matrix((real_train_size, word_embeddings_dim), dtype=np.float32)
x = sp.csr_matrix((data_x, (row_x, col_x)), shape=(
    real_train_size, word_embeddings_dim))

y = []
for i in range(real_train_size):
    # doc_meta = shuffle_doc_name_list[i]
    # temp = doc_meta.split('\t')
    # label = temp[2]
    label = shuffle_doc_label_list[i]
    # one_hot = [0 for l in range(len(label_list))]
    # label_index = label_list.index(label)
    one_hot = [0 for l in range(len(full_label))]
    label_index = full_label.index(label)
    one_hot[label_index] = 1
    y.append(one_hot)
y = np.array(y)
print(y)

# vx: for validation
val_size = len(val_ids)

row_vx = []
col_vx = []
data_vx = []
for i in range(val_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i + train_size]
    words = mecab_tagger.parse(doc_words).split()
    doc_len = len(words)
    # if doc_len <= 0:
    #     print("doc_words:", doc_words)
    #     print("doc_len:", doc_len)

    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_vx.append(i)
        col_vx.append(j)
        data_vx.append(doc_vec[j] / doc_len)

vx = sp.csr_matrix((data_vx, (row_vx, col_vx)),
                   shape=(val_size, word_embeddings_dim))

vy = []
for i in range(val_size):
    label = shuffle_doc_label_list[i + train_size]  # should be replaced with shuffled_val_label_list
    # one_hot = [0 for l in range(len(label_list))]
    # label_index = label_list.index(label)
    one_hot = [0 for l in range(len(full_label))]
    label_index = full_label.index(label)
    one_hot[label_index] = 1
    vy.append(one_hot)
vy = np.array(vy)
print(vy)

# tx: feature vectors of test docs, no initial features
test_size = len(test_ids)

row_tx = []
col_tx = []
data_tx = []
for i in range(test_size):
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i + train_size + val_size]
    words = mecab_tagger.parse(doc_words).split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_tx.append(i)
        col_tx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_tx.append(doc_vec[j] / doc_len)  # doc_vec[j] / doc_len

# tx = sp.csr_matrix((test_size, word_embeddings_dim), dtype=np.float32)
tx = sp.csr_matrix((data_tx, (row_tx, col_tx)),
                   shape=(test_size, word_embeddings_dim))

ty = []
for i in range(test_size):
    # doc_meta = shuffle_doc_name_list[i + train_size + val_size]
    # temp = doc_meta.split('\t')
    # label = temp[2]
    label = shuffle_doc_label_list[i + train_size + val_size]  # should be replaced with shuffled_test_label_list
    # one_hot = [0 for l in range(len(label_list))]
    # label_index = label_list.index(label)
    one_hot = [0 for l in range(len(full_label))]
    label_index = full_label.index(label)
    one_hot[label_index] = 1
    ty.append(one_hot)
ty = np.array(ty)
print(ty)

# allx: the the feature vectors of both labeled and unlabeled training instances
# (a superset of x)
# unlabeled training instances -> words

word_vectors = np.random.uniform(-0.01, 0.01,
                                 (vocab_size, word_embeddings_dim))

for i in range(len(vocab)):
    word = vocab[i]
    if word in word_vector_map:
        vector = word_vector_map[word]
        word_vectors[i] = vector

row_allx = []
col_allx = []
data_allx = []

for i in range(train_size + val_size):  # following the original code
    doc_vec = np.array([0.0 for k in range(word_embeddings_dim)])
    doc_words = shuffle_doc_words_list[i]
    words = mecab_tagger.parse(doc_words).split()
    doc_len = len(words)
    for word in words:
        if word in word_vector_map:
            word_vector = word_vector_map[word]
            doc_vec = doc_vec + np.array(word_vector)

    for j in range(word_embeddings_dim):
        row_allx.append(int(i))
        col_allx.append(j)
        # np.random.uniform(-0.25, 0.25)
        data_allx.append(doc_vec[j] / doc_len)  # doc_vec[j]/doc_len
for i in range(vocab_size):
    for j in range(word_embeddings_dim):
        row_allx.append(int(i + train_size + val_size))
        col_allx.append(j)
        data_allx.append(word_vectors.item((i, j)))

row_allx = np.array(row_allx)
col_allx = np.array(col_allx)
data_allx = np.array(data_allx)

allx = sp.csr_matrix(
    (data_allx, (row_allx, col_allx)), shape=(train_size + val_size + vocab_size, word_embeddings_dim))

ally = []
for i in range(train_size + val_size):
    # doc_meta = shuffle_doc_name_list[i]
    # temp = doc_meta.split('\t')
    # label = temp[2]
    label = shuffle_doc_label_list[i]
    # one_hot = [0 for l in range(len(label_list))]
    # label_index = label_list.index(label)
    one_hot = [0 for l in range(len(full_label))]
    label_index = full_label.index(label)
    one_hot[label_index] = 1
    ally.append(one_hot)

for i in range(vocab_size):
    # one_hot = [0 for l in range(len(label_list))]
    one_hot = [0 for l in range(len(full_label))]
    ally.append(one_hot)

ally = np.array(ally)

print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

'''
Doc word heterogeneous graph
'''

# word co-occurence with context windows
window_size = 20
windows = []

for doc_words in shuffle_doc_words_list:
    words = mecab_tagger.parse(doc_words).split()
    length = len(words)
    if length <= window_size:
        windows.append(words)
    else:
        # print(length, length - window_size + 1)
        for j in range(length - window_size + 1):
            window = words[j: j + window_size]
            windows.append(window)
            # print(window)

word_window_freq = {}
for window in windows:
    appeared = set()
    for i in range(len(window)):
        if window[i] in appeared:
            continue
        if window[i] in word_window_freq:
            word_window_freq[window[i]] += 1
        else:
            word_window_freq[window[i]] = 1
        appeared.add(window[i])

word_pair_count = {}
for window in windows:
    for i in range(1, len(window)):
        for j in range(0, i):
            word_i = window[i]
            word_i_id = word_id_map[word_i]
            word_j = window[j]
            word_j_id = word_id_map[word_j]
            if word_i_id == word_j_id:
                continue
            word_pair_str = str(word_i_id) + ',' + str(word_j_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1
            # two orders
            word_pair_str = str(word_j_id) + ',' + str(word_i_id)
            if word_pair_str in word_pair_count:
                word_pair_count[word_pair_str] += 1
            else:
                word_pair_count[word_pair_str] = 1

row = []
col = []
weight = []

# pmi as weights

num_window = len(windows)

if weight_mode == 'pmi':
    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[vocab[i]]
        word_freq_j = word_window_freq[vocab[j]]
        pmi = log((1.0 * count / num_window) /
                  (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue

        row.append(train_size + val_size + i)
        col.append(train_size + val_size + j)
        weight.append(pmi)

# word vector cosine similarity as weights
elif weight_mode == 'cos' or weight_mode == 'cos_cos':
    print("Using cosine similarity as weights")
    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        word_i_str = vocab[i]
        word_j_str = vocab[j]
        sim = cos_similarity(word_i_str, word_j_str, ft)

        if sim <= 0:
            continue
        row.append(train_size + val_size + i)
        col.append(train_size + val_size + j)
        weight.append(sim)

# word vector cosine similarity as weights

'''
for i in range(vocab_size):
    for j in range(vocab_size):
        if vocab[i] in word_vector_map and vocab[j] in word_vector_map:
            vector_i = np.array(word_vector_map[vocab[i]])
            vector_j = np.array(word_vector_map[vocab[j]])
            similarity = 1.0 - cosine(vector_i, vector_j)
            if similarity > 0.9:
                print(vocab[i], vocab[j], similarity)
                row.append(train_size + i)
                col.append(train_size + j)
                weight.append(similarity)
'''
# doc word frequency
doc_word_freq = {}

for doc_id in range(len(shuffle_doc_words_list)):
    doc_words = shuffle_doc_words_list[doc_id]
    words = mecab_tagger.parse(doc_words).split()
    for word in words:
        # print("DEBUG word", word)
        word_id = word_id_map[word]
        doc_word_str = str(doc_id) + ',' + str(word_id)
        if doc_word_str in doc_word_freq:
            doc_word_freq[doc_word_str] += 1
        else:
            doc_word_freq[doc_word_str] = 1

if weight_mode != 'cos_cos':
    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = mecab_tagger.parse(doc_words).split()
        doc_word_set = set()
        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            key = str(i) + ',' + str(j)
            freq = doc_word_freq[key]
            if i < (train_size + val_size):
                row.append(i)
            else:
                row.append(i + vocab_size)
            col.append(train_size + val_size + j)
            idf = log(1.0 * len(shuffle_doc_words_list) /
                      word_doc_freq[vocab[j]])
            weight.append(freq * idf)
            doc_word_set.add(word)

elif weight_mode == 'cos_cos':
    for i in range(len(shuffle_doc_words_list)):
        doc_words = shuffle_doc_words_list[i]
        words = mecab_tagger.parse(doc_words).split()
        doc_word_set = set()

        for word in words:
            if word in doc_word_set:
                continue
            j = word_id_map[word]
            if i < (train_size + val_size):
                row.append(i)
            else:
                row.append(i + vocab_size)
            col.append(train_size + val_size + j)
            sim = doc_word_similarity(word, words, ft)
            weight.append(sim)
            doc_word_set.add(word)

node_size = train_size + val_size + vocab_size + test_size
adj = sp.csr_matrix(
    (weight, (row, col)), shape=(node_size, node_size))

# dump objects
f = open("data/ind.{}.{}.x".format(dataset, weight_mode), 'wb')
pkl.dump(x, f)
f.close()

f = open("data/ind.{}.{}.y".format(dataset, weight_mode), 'wb')
pkl.dump(y, f)
f.close()

f = open("data/ind.{}.{}.tx".format(dataset, weight_mode), 'wb')
pkl.dump(tx, f)
f.close()

f = open("data/ind.{}.{}.ty".format(dataset, weight_mode), 'wb')
pkl.dump(ty, f)
f.close()

f = open("data/ind.{}.{}.allx".format(dataset, weight_mode), 'wb')
pkl.dump(allx, f)
f.close()

f = open("data/ind.{}.{}.ally".format(dataset, weight_mode), 'wb')
pkl.dump(ally, f)
f.close()

f = open("data/ind.{}.{}.adj".format(dataset, weight_mode), 'wb')
pkl.dump(adj, f)
f.close()
