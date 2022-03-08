import os
import random
import fasttext
import fasttext.util
import scipy

import konlpy
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from konlpy.tag import Hannanum
from konlpy.tag import Komoran
import mecab  # python-mecab-ko
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

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'korean', 'kr_full_label']

# fasttext download model
fasttext.util.download_model('ko', if_exists='ignore')  # korean model
ft = fasttext.load_model('cc.ko.300.bin')

mecabko = mecab.MeCab()
mecabjp = MeCab.Tagger("-Owakati")

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


def word_segmenting(doc, lang):
    if lang == 'korean':
        words = mecabko.morphs(doc)
    elif lang == 'japanese':
        words = mecabjp.parse(doc).split()
    else:
        words = doc.split()
    return words


# build corpus
dataset = sys.argv[1]
weight_mode = sys.argv[2]  # ['pmi', 'cos']
# dataset = 'korean'

full_label = {
    '문학, 책': 0,
    '영화': 1,
    '미술, 디자인': 2,
    '공연, 전시': 3,
    '음악': 4,
    '드라마': 5,
    '스타, 연예인': 6,
    '만화, 애니': 7,
    '방송': 8,
    '일상, 생각': 9,
    '육아, 결혼': 10,
    '애완, 반려동물': 11,
    '좋은글, 이미지': 12,
    '패션, 미용': 13,
    '인테리어, DIY': 14,
    '요리, 레시피': 15,
    '상품리뷰': 16,
    '원예, 재배': 17,
    '게임': 18,
    '스포츠': 19,
    '사진': 20,
    '자동차': 21,
    '취미': 22,
    '국내여행': 23,
    '세계여행': 24,
    '맛집': 25,
    'IT, 컴퓨터': 26,
    '사회, 정치': 27,
    '건강, 의학': 28,
    '비즈니스, 경제': 29,
    '어학, 외국어': 30,
    '교육, 학문': 31,
    'literature, books': 0,
    'movie': 1,
    'art, design': 2,
    'performance, exhibition': 3,
    'music': 4,
    'drama': 5,
    'star, celebrity': 6,
    'cartoon, anime': 7,
    'broadcast': 8,
    'everyday, thoughts': 9,
    'parenting, marriage': 10,
    'pets, companion animal': 11,
    'good article, image': 12,
    'fashion, beauty': 13,
    'interior, DIY': 14,
    'cooking recipe': 15,
    'product review': 16,
    'horticulture, cultivation': 17,
    'game': 18,
    'sports': 19,
    'picture': 20,
    'car': 21,
    'hobby': 22,
    'domestic travel': 23,
    'world travel': 24,
    'restaurant': 25,
    'IT, computer': 26,
    'society, politics': 27,
    'health, medicine': 28,
    'business, economy': 29,
    'language, foreign language': 30,
    'education, academic': 31,
    '文学、本': 0,
    '映画': 1,
    '芸術、デザイン': 2,
    'パフォーマンス、ショー': 3,
    '音楽': 4,
    'ドラマ': 5,
    'スター、有名人': 6,
    'アニメ': 7,
    'テレビ、ラジオ放送': 8,
    '日常': 9,
    '子育て・結婚': 10,
    'ペット': 11,
    '良い記事、画像': 12,
    'ファッション、美容': 13,
    'インテリア、DIY': 14,
    '料理レシピ': 15,
    '製品レビュー': 16,
    '園芸、栽培': 17,
    'ゲーム': 18,
    'スポーツ': 19,
    '写真': 20,
    '車': 21,
    '趣味': 22,
    '国内旅行': 23,
    '海外旅行': 24,
    'レストラン': 25,
    'IT、コンピュータ': 26,
    '社会、政治': 27,
    '健康、医学': 28,
    'ビジネス、経済': 29,
    '言語、外国語': 30,
    '教育、学術': 31
}

# if dataset not in datasets:
#     sys.exit("wrong dataset name")

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

label_set = set()
train_label_list = []
val_label_list = []
test_label_list = []
doc_label_list = []

korean_path = '/home/lr/kwonjingun/data_server/NAVER_MTC/naver/dataset/processed/both/'
japanese_path = 'data/japanese/'
english_path = '/home/lr/kwonjingun/data_server/NAVER_MTC/naver/reddit_dataset/annotated/splited/'
lang_paths = [korean_path, japanese_path, english_path]
langs = ['korean', 'japanese', 'english']

for lang, lang_path in zip(langs, lang_paths):
    splitter = '\t' if lang_path == japanese_path else '</s>'

    ftrain = open(lang_path + 'train.source', 'r', encoding='utf-8')
    fval = open(lang_path + 'val.source', 'r', encoding='utf-8')
    ftest = open(lang_path + 'test.source', 'r', encoding='utf-8')

    lines = ftrain.readlines()
    source_train_idx = []
    source_val_idx = []
    source_test_idx = []
    for i, line in enumerate(lines):
        line = line.strip()
        title, content = line.split(splitter)
        if len(content) > 0:
            source_train_idx.append(i)
            doc_name_list.append(title)
            doc_train_list.append(title)
            doc_content_list.append(word_segmenting(content, lang))
            train_content_list.append(word_segmenting(content, lang))
    ftrain.close()

    lines = fval.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        title, content = line.split(splitter)
        if len(content) > 0:
            source_val_idx.append(i)
            doc_name_list.append(title)
            doc_val_list.append(title)
            doc_content_list.append(word_segmenting(content, lang))
            val_content_list.append(word_segmenting(content, lang))
    fval.close()

    lines = ftest.readlines()
    for i, line in enumerate(lines):
        line = line.strip()
        title, content = line.split(splitter)
        if len(content) > 0:
            source_test_idx.append(i)
            doc_name_list.append(title)
            doc_test_list.append(title)
            doc_content_list.append(word_segmenting(content, lang))
            test_content_list.append(word_segmenting(content, lang))
    ftest.close()

    # for label

    ftrain = open(lang_path + 'train.target', 'r', encoding='utf-8')
    fval = open(lang_path + 'val.target', 'r', encoding='utf-8')
    ftest = open(lang_path + 'test.target', 'r', encoding='utf-8')

    lines = ftrain.readlines()
    for i, line in enumerate(lines):
        if i in source_train_idx:
            temp = line.split(">")
            # if temp[0].strip() in full_label:
            #     label_set.add(temp[0].strip())
            train_label_list.append(temp[0].strip())
            doc_label_list.append(temp[0].strip())
    ftrain.close()

    lines = fval.readlines()
    for i, line in enumerate(lines):
        if i in source_val_idx:
            temp = line.split(">")
            # if temp[0].strip() in full_label:
            #     label_set.add(temp[0].strip())
            val_label_list.append(temp[0].strip())
            doc_label_list.append(temp[0].strip())
    fval.close()

    lines = ftest.readlines()
    for i, line in enumerate(lines):
        if i in source_test_idx:
            temp = line.split(">")
            # if temp[0].strip() in full_label:
            #     label_set.add(temp[0].strip())
            test_label_list.append(temp[0].strip())
            doc_label_list.append(temp[0].strip())
    ftest.close()

# partial labeled dta
# train_ids = train_ids[:int(0.2 * len(train_ids))]
train_ids = []
for train_name in doc_train_list:
    train_id = doc_name_list.index(train_name)
    if doc_label_list[train_id] in full_label:
        train_ids.append(train_id)
# print(train_ids)
random.shuffle(train_ids)

train_ids_str = '\n'.join(str(index) for index in train_ids)
f = open('data/' + dataset + '.' + weight_mode + '.train.index', 'w', encoding='utf-8')
f.write(train_ids_str)
f.close()

val_ids = []
for val_name in doc_val_list:
    val_id = doc_name_list.index(val_name)
    if doc_label_list[val_id] in full_label:
        val_ids.append(val_id)
# print(val_ids)
random.shuffle(val_ids)

val_ids_str = '\n'.join(str(index) for index in val_ids)
f = open('data/' + dataset + '.' + weight_mode + '.val.index', 'w', encoding='utf-8')
f.write(val_ids_str)
f.close()

test_ids = []
for test_name in doc_test_list:
    test_id = doc_name_list.index(test_name)
    if doc_label_list[test_id] in full_label:
        test_ids.append(test_id)
# print(test_ids)
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
    if doc_label_list[int(idx)] in full_label:
        shuffle_doc_name_list.append(doc_name_list[int(idx)])
        if len(doc_content_list[int(idx)]) < 1:
            print("DEBUG doc_content_list")
            print(doc_content_list[int(idx)])
            print("title", str(doc_name_list[int(idx)]))
            exit()
        shuffle_doc_words_list.append(doc_content_list[int(idx)])
        shuffle_doc_label_list.append(doc_label_list[int(idx)])
shuffle_doc_name_str = '\n'.join(shuffle_doc_name_list)
shuffle_doc_words_str = '\n'.join([' '.join(doc_words) for doc_words in shuffle_doc_words_list])
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
# mecabko = mecab.MeCab()
# mecabjp = MeCab.Tagger("-Owakati")
for doc_words in shuffle_doc_words_list:
    # words = doc_words.split()
    # words = mecabko.morphs(doc_words)
    for word in doc_words:
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
    # words = doc_words.split()
    # words = hannanum.morphs(doc_words)
    # words = mecabko.morphs(doc_words)
    words = doc_words
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

# label list
# label_set = set()
# for doc_meta in shuffle_doc_name_list:
#     temp = doc_meta.split('\t')
#     label_set.add(temp[2])
# label_list = list(label_set)

# label_list_str = '\n'.join(label_list)
label_list_str = '\n'.join(full_label.keys())
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
    # words = doc_words.split()
    # words = hannanum.morphs(doc_words)
    # words = mecabko.morphs(doc_words)
    words = doc_words
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
    one_hot = [0 for l in range(int(len(full_label) / len(lang_paths)))]
    # label_index = full_label.index(label)
    label_index = full_label[label]
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
    # words = hannanum.morphs(doc_words)
    # words = mecabko.morphs(doc_words)
    words = doc_words
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
    one_hot = [0 for l in range(int(len(full_label) / len(lang_paths)))]
    # label_index = label_list.index(label)
    # label_index = full_label.index(label)
    label_index = full_label[label]
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
    # words = doc_words.split()
    # words = hannanum.morphs(doc_words)
    # words = mecabko.morphs(doc_words)
    words = doc_words
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
    one_hot = [0 for l in range(int(len(full_label) / len(lang_paths)))]
    # label_index = full_label.index(label)
    label_index = full_label[label]
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
    # words = doc_words.split()
    # words = hannanum.morphs(doc_words)
    # words = mecabko.morphs(doc_words)
    words = doc_words
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
    one_hot = [0 for l in range(int(len(full_label) / len(lang_paths)))]
    # label_index = full_label.index(label)
    label_index = full_label[label]
    one_hot[label_index] = 1
    ally.append(one_hot)

for i in range(vocab_size):
    # one_hot = [0 for l in range(len(label_list))]
    one_hot = [0 for l in range(int(len(full_label) / len(lang_paths)))]
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
    # words = doc_words.split()
    # words = hannanum.morphs(doc_words)
    # words = mecabko.morphs(doc_words)
    words = doc_words
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
# option 2: cos_similarity as weights

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
    # words = hannanum.morphs(doc_words)
    # words = mecabko.morphs(doc_words)
    # words = doc_words.split()
    words = doc_words
    # print("DEBUG word_id_map", len(word_id_map))
    # print("DEBUG len vocab", len(vocab))
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
        # words = doc_words.split()
        # words = hannanum.morphs(doc_words)
        # words = mecabko.morphs(doc_words)
        words = doc_words
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
        # words = mecabko.morphs(doc_words)
        words = doc_words
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
