# -*- coding: utf-8 -*-
# @Time    : 2023/11/7 下午 8:13
# @Author  : Guoshuai Zhang
# @E-mail  : guoshuai0914(a)126.com
# @File    : topic_tracking_plot.py
# @Software: PyCharm
# Copyright(c) 2023, Guoshuai Zhang. All Rights Reserved.


import numpy as np
from scipy.optimize import linear_sum_assignment
import pickle
from datetime import datetime, timedelta
from collections import Counter
import operator
import matplotlib.pyplot as plt
import csv


def hungarian_algorithm(cost_matrix):
    index_list = list()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    for i in range(len(row_ind)):
        index_list.append((row_ind[i], col_ind[i]))

    return index_list


def make_matrix(r, c, initial, num_type):
    a_score = np.zeros(shape=[r, c], dtype=num_type)
    for i in range(r):
        for j in range(c):
            a_score[i, j] = initial
    return a_score


def diag_matrix(matrix, diag):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if i == j: matrix[i, j] = diag
    return matrix


with open('./all_topics_20230713_1207_19741953.pkl', 'rb') as topic_f:
    topics_dict = pickle.load(topic_f)
# pre-process dict
del_keys = list()
for k_tmp in topics_dict.keys():
    if len(topics_dict[k_tmp]) == 0:
        del_keys.append(k_tmp)
for k_del in del_keys:
    del topics_dict[k_del]


# 1) associate topics
collection_a = list()

for k_inter_d, date_id in enumerate(list(topics_dict.keys())):
    if k_inter_d == 0:
        collection_a = ['%s_%s' % (date_id, topic_id) for topic_id in list(topics_dict[date_id].keys())]
    else:
        collection_b = ['%s_%s' % (date_id, topic_id) for topic_id in list(topics_dict[date_id].keys())]

        assign_score = make_matrix(len(collection_a), len(collection_b), 100, np.float32)
        for a_i in range(len(collection_a)):
            latest_a_topic_index = collection_a[a_i].split('+')[-1]
            latest_a_topic_id = int(latest_a_topic_index.split('_')[-1])
            latest_a_topic_date = latest_a_topic_index.split('_')[0]
            a_embedding = topics_dict[latest_a_topic_date][latest_a_topic_id]['topic_embedding']

            for b_j in range(len(collection_b)):
                latest_b_topic_index = collection_b[b_j]
                latest_b_topic_id = int(latest_b_topic_index.split('_')[-1])
                latest_b_topic_date = latest_b_topic_index.split('_')[0]
                b_embedding = topics_dict[latest_b_topic_date][latest_b_topic_id]['topic_embedding']

                a_score = np.sqrt(np.mean(np.square(a_embedding - b_embedding)))
                # print(a_score)
                if a_score > 0.02:
                    a_score = 1
                assign_score[a_i, b_j] = a_score

        # 2) Appearance score
        left_top = assign_score

        m = len(collection_a)
        n = len(collection_b)
        right_top = diag_matrix(matrix=make_matrix(m, m, 100, num_type=np.float32), diag=0.015)
        left_down = diag_matrix(matrix=make_matrix(n, n, 100, num_type=np.float32), diag=0.015)
        right_down = np.zeros(shape=[n, m], dtype=np.float32)
        sub1 = np.concatenate((left_top, right_top), axis=1)
        sub2 = np.concatenate((left_down, right_down), axis=1)
        cost_score = np.concatenate((sub1, sub2), axis=0)

        match_list = hungarian_algorithm(cost_matrix=cost_score)

        b_pop_list = list()
        for match_a_r, match_a_c in match_list:
            if match_a_r < len(collection_a) and match_a_c < len(collection_b):
                collection_a[match_a_r] = collection_a[match_a_r] + '+' + collection_b[match_a_c]
                b_pop_list.append(collection_b[match_a_c])

        for b_pop_l in b_pop_list:
            collection_b.remove(b_pop_l)
        if len(collection_b) == 0:
            continue
        else:
            for remain_b in collection_b:
                collection_a.append(remain_b)

# 1.1) filter collection a
collection_a_del = list()
for sub_a in collection_a:
    if len(sub_a.split('+')) <= 2:
        collection_a_del.append(sub_a)
    else:
        continue
for del_a in collection_a_del:
    collection_a.remove(del_a)
print(len(collection_a))

# 2) show topics tracklet
start_datetime = datetime.strptime('2023-07-13', '%Y-%m-%d')
end_datetime = datetime.strptime('2023-12-07', '%Y-%m-%d')
days_len = (end_datetime - start_datetime).days + 1

topic_tracklet = collection_a
tracklet_dic = dict()
for series_index, topics_series in enumerate(topic_tracklet):
    tracklet_dic['topic_tracklet_%d' % series_index] = {'occurrence_count': [0] * days_len, 'topic_keywords': 'ini'}
    keywords_collection = list()
    date_list = list()

    topic_element_dic = dict()
    for topic_element in topics_series.split('+'):
        sub_date = topic_element.split('_')[0]
        sub_topic_id = int(topic_element.split('_')[-1])
        topic_element_dic[sub_date] = sub_topic_id

    for day_index, _ in enumerate(range(days_len)):
        current_date = datetime.strftime(start_datetime + timedelta(days=day_index), '%Y-%m-%d')
        date_list.append(current_date)

        # a) add keywords occurrence count
        if current_date in list(topic_element_dic.keys()):
            tracklet_dic['topic_tracklet_%d' % series_index]['occurrence_count'][day_index] += \
                topics_dict[current_date][topic_element_dic[current_date]]['count']

            # b.0) merge keywords
            for w in topics_dict[current_date][topic_element_dic[current_date]]['topic_words']:
                keywords_collection.append(w)

        else:
            continue

    # b.1) merge keywords
    merge_keywords = '_'.join([ck_pair[0] for ck_pair in sorted(Counter(keywords_collection).items(),
                                                                key=operator.itemgetter(1), reverse=True)[:5]])
    tracklet_dic['topic_tracklet_%d' % series_index]['topic_keywords'] = merge_keywords

# 3) plot & save the result
plt.subplots()  # figsize=(5, 2.7), constrained_layout=True
save_file = open('./topic_tracklet_file.csv', 'w', encoding='utf-8', newline='')
writer = csv.writer(save_file)
writer.writerow(['id', 'topic'] + date_list)
for tracklet_ex in tracklet_dic.keys():
    print(datetime.strftime(datetime.utcnow(), '%Y-%m-%d %H:%M:%S'), '[plot] %s' % tracklet_ex)
    tracklet_id_ = tracklet_ex.split('_')[-1]
    occurrence_freq = tracklet_dic[tracklet_ex]['occurrence_count']
    tracklet_name = tracklet_dic[tracklet_ex]['topic_keywords']
    writer.writerow([tracklet_id_, tracklet_name] + occurrence_freq)

    plt.plot(date_list, occurrence_freq, label=tracklet_name)
    plt.xticks(range(1, len(date_list), 15))
    # ax.set_xlabel('')
    plt.ylabel('the number of usages of the topic')
    # ax.set_title('')
    # plt.legend()
    plt.tight_layout()

plt.savefig('./tracking_topics.png', dpi=500, bbox_inches='tight')
plt.show()
plt.close()
plt.close('all')
save_file.close()


