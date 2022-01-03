import MeCab
import os
import random

from collections import Counter


def write_file(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        for datum in data:
            f.write(datum+'\n')


tagger = MeCab.Tagger('-Owakati')
# counter = Counter()

file_path = '/home/lr/kamigaito/Data/Corpora/Naver/HierTopic/Ja.tsv'
save_path = 'data/japanese/'
fraw = open(file_path, 'r', encoding='utf-8')

val_test_ratio = 0.1

lines = fraw.readlines()

preprocessed_data = []
preprocessed_source = []
preprocessed_target = []

for line in lines:
    counter = Counter()
    title, document, topic1, topic2, topic3, topic4, topic5, batch_id = line.split('\t')
    topics = [topic1, topic2, topic3, topic4, topic5]
    topics = [topic.split('>')[0] for topic in topics]

    for topic in topics:
        counter[topic] += 1

    top_topic = counter.most_common(1)
    if top_topic[0][1] > 1:
        temp = title + "\t" + document + "\t" + top_topic[0][0]
        temp_source = title + "\t" + document
        temp_target = top_topic[0][0]
        preprocessed_data.append(temp)
        preprocessed_source.append(temp_source)
        preprocessed_target.append(temp_target)
fraw.close()

# random.shuffle(preprocessed_data)
total_length = len(preprocessed_data)
split_point = int(val_test_ratio * total_length)
test_data, val_data, train_data = preprocessed_data[:split_point], preprocessed_data[split_point:split_point*2], \
                                  preprocessed_data[split_point*2:]
test_source, val_source, train_source = preprocessed_source[:split_point], \
                                        preprocessed_source[split_point:split_point*2], \
                                        preprocessed_source[split_point*2:]
test_target, val_target, train_target = preprocessed_target[:split_point], \
                                        preprocessed_target[split_point:split_point*2], \
                                        preprocessed_target[split_point*2:]

# write both source and target in the same file
write_file(save_path+'train_data.tsv', train_data)
write_file(save_path+'test_data.tsv', test_data)
write_file(save_path+'val_data.tsv', val_data)

# write source and target in seperate files
write_file(save_path+'train.source', train_source)
write_file(save_path+'test.source', test_source)
write_file(save_path+'val.source', val_source)

write_file(save_path+'train.target', train_target)
write_file(save_path+'val.target', val_target)
write_file(save_path+'test.target', test_target)






