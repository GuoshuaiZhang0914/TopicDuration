# -*- coding: utf-8 -*-
# @Time    : 2023/12/11 下午 2:44
# @Author  : Guoshuai Zhang
# @E-mail  : guoshuai0914(a)126.com
# @File    : topic_tracking_properties.py
# @Software: PyCharm
# Copyright(c) 2023, Guoshuai Zhang. All Rights Reserved.


import numpy as np
from scipy.optimize import linear_sum_assignment
import pickle
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import networkx as nx
import csv
import emoji
import pandas as pd


def judge_date_range(date_point):
    target_date = datetime.strptime(date_point[:10], '%Y-%m-%d')

    if datetime.strptime('2023-07-13', '%Y-%m-%d') <= target_date <= datetime.strptime('2023-08-12', '%Y-%m-%d'):
        return '2023-07-13+2023-08-12'
    elif datetime.strptime('2023-08-13', '%Y-%m-%d') <= target_date <= datetime.strptime('2023-09-12', '%Y-%m-%d'):
        return '2023-08-13+2023-09-12'
    elif datetime.strptime('2023-09-13', '%Y-%m-%d') <= target_date <= datetime.strptime('2023-10-12', '%Y-%m-%d'):
        return '2023-09-13+2023-10-12'
    elif datetime.strptime('2023-10-13', '%Y-%m-%d') <= target_date <= datetime.strptime('2023-11-12', '%Y-%m-%d'):
        return '2023-10-13+2023-11-12'
    else:
        return '2023-11-13+2023-12-12'


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


def properties_for_topic(all_topics, topic_tracklet):
    t2 = 0
    t32 = 0
    t33 = 0
    ts_count = list()
    count_ = list()
    for topic_element in topic_tracklet.split('+'):
        sub_date = topic_element.split('_')[0]
        sub_topic_id = int(topic_element.split('_')[-1])
        ts_count.append(all_topics[sub_date][sub_topic_id]['topic_ts'][0])
        count_.append(all_topics[sub_date][sub_topic_id]['count'])
        t2 += all_topics[sub_date][sub_topic_id]['count']

        for topic_sen in all_topics[sub_date][sub_topic_id]['topic_content']:
            t32 += len(topic_sen.split())
            t33 += sum([topic_sen.count(emoji_) for emoji_ in emoji.EMOJI_DATA])

    topic_start_date = datetime.strptime(ts_count[0][:-11], '%Y-%m-%d %H:%M:%S')
    topic_end_date = datetime.strptime(all_topics[sub_date][sub_topic_id]['topic_ts'][-1][:-11], '%Y-%m-%d %H:%M:%S')

    if topic_end_date == topic_start_date and len(all_topics[sub_date][sub_topic_id]['topic_ts']) > 1:
        topic_end_date = datetime.strptime(all_topics[sub_date][sub_topic_id]['topic_ts'][-2][:-11], '%Y-%m-%d %H:%M:%S')
    elif topic_end_date == topic_start_date and len(all_topics[sub_date][sub_topic_id]['topic_ts']) == 1:
        topic_end_date = topic_end_date + timedelta(minutes=30)

    topic_duration_sec = (topic_end_date - topic_start_date).total_seconds()
    topic_duration_day = round((topic_end_date - topic_start_date).days +
                               (topic_end_date - topic_start_date).seconds/86400, 2)
    first_peak_sec = (datetime.strptime(ts_count[count_.index(max(count_))][:-11], '%Y-%m-%d %H:%M:%S') -
                      topic_start_date).total_seconds()

    return topic_duration_sec, round(t2/topic_duration_day, 2), t2, first_peak_sec, round(t32/t2, 2), round(t33/t2, 2)


def influence_of_sponsor(all_topics, topic_tracklet, user_info):
    p12 = 0
    latest_subtopic_date = topic_tracklet.split('+')[0].split('_')[0]
    latest_subtopic_id = int(topic_tracklet.split('+')[0].split('_')[-1])
    latest_subtopic_uid = all_topics[latest_subtopic_date][latest_subtopic_id]['topic_uid'][0]
    latest_subtopic_time = all_topics[latest_subtopic_date][latest_subtopic_id]['topic_ts'][0]

    for sub_date in all_topics.keys():
        for sub_topic_id in all_topics[sub_date].keys():
            for uid_ in all_topics[sub_date][sub_topic_id]['topic_uid']:
                if uid_ == latest_subtopic_uid:
                    p12 += 1

    target_date_point = judge_date_range(date_point=latest_subtopic_time)
    p11 = user_info[target_date_point][latest_subtopic_uid]['is_union_leader']
    p131 = user_info[target_date_point][latest_subtopic_uid]['user_power']
    p132 = user_info[target_date_point][latest_subtopic_uid]['user_strength']
    p133 = user_info[target_date_point][latest_subtopic_uid]['user_level']
    p14 = user_info[target_date_point][latest_subtopic_uid]['user_time']  # minute

    # p11, p133, p131, p132, p14 = get_user_info(user_id=latest_subtopic_uid, query_time=latest_subtopic_time)

    return p11, p12, p131, p132, p133, p14 * 60


def get_user_level(_uid_, _date_, _usr_info_):
    target_date_point = judge_date_range(date_point=_date_)

    return _usr_info_[target_date_point][_uid_]['user_level']


def influence_of_opinion_leader(all_topics, topic_tracklet, user_info):
    uid_dic = dict()
    other_topic_user = list()
    other_sen_count = 0
    opinion_leader = list()
    opinion_sen_count = 0

    for topic_element in topic_tracklet.split('+'):
        sub_date = topic_element.split('_')[0]
        sub_topic_id = int(topic_element.split('_')[-1])

        for uid_ in all_topics[sub_date][sub_topic_id]['topic_uid'][1:]:
            if uid_ not in list(uid_dic.keys()):
                uid_dic[uid_] = {'count': 1, 'user_level': get_user_level(_uid_=uid_, _date_=sub_date,
                                                                          _usr_info_=user_info)}
            else:
                uid_dic[uid_]['count'] += 1

    for user_k in uid_dic.keys():
        if uid_dic[user_k]['user_level'] >= 35:
            opinion_leader.append(user_k)
            opinion_sen_count += uid_dic[user_k]['count']
        else:
            other_topic_user.append(user_k)
            other_sen_count += uid_dic[user_k]['count']

    return round(len(opinion_leader) / len(list(uid_dic.keys())), 2), opinion_sen_count


def influence_of_participator(all_topics, topic_tracklet, user_info):
    p321 = list()
    p322 = list()
    p323 = list()
    p33 = list()

    uid_dic = dict()
    for topic_element in topic_tracklet.split('+'):
        sub_date = topic_element.split('_')[0]
        sub_topic_id = int(topic_element.split('_')[-1])
        date_range = judge_date_range(sub_date)

        for uid_ in all_topics[sub_date][sub_topic_id]['topic_uid'][1:]:
            power_ = user_info[date_range][uid_]['user_power']
            strength_ = user_info[date_range][uid_]['user_strength']
            level_ = user_info[date_range][uid_]['user_level']
            time_ = user_info[date_range][uid_]['user_time']

            if uid_ not in list(uid_dic.keys()):
                uid_dic[uid_] = {'count': 1, 'power': [power_],
                                 'strength': [strength_],
                                 'level': [level_],
                                 'online_time': [time_]}
            else:
                uid_dic[uid_]['count'] += 1
                uid_dic[uid_]['power'].append(power_)
                uid_dic[uid_]['strength'].append(strength_)
                uid_dic[uid_]['level'].append(level_)
                uid_dic[uid_]['online_time'].append(time_)

    p31 = sum([uid_dic[u]['count'] for u in uid_dic.keys()])
    for u in uid_dic.keys():
        p321 = p321 + uid_dic[u]['power']
        p322 = p322 + uid_dic[u]['strength']
        p323 = p323 + uid_dic[u]['level']
        p33 = p33 + uid_dic[u]['online_time']

    return p31, round(sum(p321)/len(p321), 2), round(sum(p322)/len(p322), 2), round(sum(p323)/len(p323), 2), \
        round(sum(p33)/len(p33), 2)


print('[0/3] Reading and  pre-processing data ...')
with open('./all_topics_20230713_1207_19741953.pkl', 'rb') as topic_f:
    topics_dict = pickle.load(topic_f)
# remove AI Chatbot 'uid=6'
for d_k in topics_dict.keys():
    for t_k in topics_dict[d_k].keys():
        del_user = list()  # uid=6
        for u_id_in, u_id in enumerate(topics_dict[d_k][t_k]['topic_uid']):
            if len(u_id) < 5:
                del_user.append(u_id_in)
        del_user.sort(reverse=True)
        for del_user_i in del_user:
            print('Delete element in DATE:%s, TOPIC:%d' % (d_k, t_k))
            del topics_dict[d_k][t_k]['topic_content'][del_user_i]
            del topics_dict[d_k][t_k]['topic_ts'][del_user_i]
            del topics_dict[d_k][t_k]['topic_uid'][del_user_i]
            topics_dict[d_k][t_k]['count'] -= 1
# pre-process dict
del_keys = list()
for k_tmp in topics_dict.keys():
    if len(topics_dict[k_tmp]) == 0:
        del_keys.append(k_tmp)
for k_del in del_keys:
    del topics_dict[k_del]

del_keys = list()
for k_tmp in topics_dict.keys():
    date_list_ = list(topics_dict[k_tmp].keys())
    date_list_.reverse()
    for t_tmp in date_list_:
        if len(topics_dict[k_tmp][t_tmp]['topic_uid']) == 0:
            del_keys.append([k_tmp, t_tmp])
for k_del in del_keys:
    del topics_dict[k_del[0]][k_del[1]]

# count user
tmp_user = list()
for tmp_d in topics_dict.keys():
    for tmp_id in topics_dict[tmp_d].keys():
        for tmp_uid in topics_dict[tmp_d][tmp_id]['topic_uid']:
            if tmp_uid not in tmp_user:
                tmp_user.append(tmp_uid)
            else:
                continue
print('User number: %d' % len(tmp_user))

with open('./all_user_info.pkl', 'rb') as usrinfo_f:
    usr_info = pickle.load(usrinfo_f)
# # pre-process dict
# for usr_k in usr_info.keys():
#     usr_file = pd.read_csv('./%s_userisunionleader.csv' % usr_k, encoding='utf-8')
#     usr_list = list()
#     for usr_row in usr_file.iterrows():
#         usr_list.append(str(usr_row[-1][-1]))
#     for leader_k in usr_info[usr_k].keys():
#         if leader_k in usr_list:
#             continue
#         else:
#             usr_info[usr_k][leader_k]['is_union_leader'] = 0
# usr_info['2023-09-13+2023-11-12'] = usr_info['2023-10-13+2023-11-12']
# del usr_info['2023-10-13+2023-11-12']

# 1) Tracking topics
collection_a = list()
print('[1/3] Associating topics ...')
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

# 2) Topic-User Graph
print('[2/3] Build graph ...')
Nodes = {'topic_nodes': dict(), 'user_nodes': dict()}  # node:size
Edges = dict()  # edge:width
for date_k in topics_dict.keys():
    for topic_k in topics_dict[date_k].keys():
        topic_name = '%s_%d' % (date_k, topic_k)

        # A) update topic nodes
        if topic_name not in list(Nodes['topic_nodes'].keys()):
            Nodes['topic_nodes'][topic_name] = topics_dict[date_k][topic_k]['count']
        else:
            Nodes['topic_nodes'][topic_name] += topics_dict[date_k][topic_k]['count']
        # B) update user nodes and edges
        for u_id in topics_dict[date_k][topic_k]['topic_uid']:
            # B.1) for user nodes
            if u_id not in list(Nodes['user_nodes'].keys()):
                Nodes['user_nodes'][u_id] = 1
            # else:
            #     Nodes['user_nodes'][u_id] += 1
            # B.2) for edges
            edge_name = '%s+%s' % (topic_name, u_id)
            if edge_name not in list(Edges.keys()):
                Edges[edge_name] = 1
            else:
                Edges[edge_name] += 1

topic_nodes = [tn for tn in Nodes['topic_nodes'].keys()]
user_nodes = [int(un) for un in Nodes['user_nodes'].keys()]
node_size = [Nodes['topic_nodes'][tn] for tn in Nodes['topic_nodes'].keys()] + \
            [Nodes['user_nodes'][un] for un in Nodes['user_nodes'].keys()]
user_topic_edges = [(e.split('+')[0], int(e.split('+')[-1])) for e in Edges.keys()]
edge_width = [int(Edges[e]) for e in Edges.keys()]
node_color = ['green'] * len(topic_nodes) + ['darkorange'] * len(user_nodes)
# purple: 'indigo'

# plt.figure(figsize=(10, 6.18))
G = nx.Graph()
G.add_nodes_from(topic_nodes)
G.add_nodes_from(user_nodes)
G.add_edges_from(user_topic_edges)
nx.draw(G,
        node_size=[n_z if n_z_index in range(len(topic_nodes)) else n_z * 50 for n_z_index, n_z in enumerate(node_size)],
        width=[((e_w-min(edge_width))/(max(edge_width) - min(edge_width)))*2 for e_w in edge_width],
        node_color=node_color, with_labels=False, font_size=7)
plt.savefig('./topic_graph.png', dpi=1200, bbox_inches='tight')
plt.show()


# 3) Compute and Save Properties
print('[3/3] Compute and save properties ...')
save_file = open('./topic_properties_file.csv', 'w', encoding='utf-8', newline='')
writer = csv.writer(save_file)
writer.writerow(['topic_tracklet_id'] +
                ['D', 'T1', 'T2', 'T31', 'T32', 'T33',
                 'P11', 'P12', 'P131', 'P132', 'P133', 'P14',
                 'P21', 'P22',
                 'P31', 'P321', 'P322', 'P323', 'P33',
                 'N11', 'N12'])

for series_index, topics_series in enumerate(collection_a):
    # 3.0) Network Properties: N11, N12
    print('\t[Network Properties] ---> %d/%s' % (series_index, topics_series))
    N11 = round(np.mean([dict(G.degree())[sub_topic_0] for sub_topic_0 in topics_series.split('+')]), 2)
    n12 = list()
    for sub_t_in, sub_t_0 in enumerate(topics_series.split('+')):
        for sub_t_1 in topics_series.split('+')[sub_t_in + 1:]:
            try:
                n_dis = nx.shortest_path_length(G, sub_t_0, sub_t_1)

            except BaseException as err:
                print(datetime.strftime(datetime.utcnow(), '%Y-%m-%d %H:%M:%S'), '[Translate API ERROR:]', err)
                n_dis = float('inf')

            n12.append(n_dis)
    N12 = round(np.mean(n12), 2)

    # 3.1) Topic Properties: D(seconds), T1, T2, T31, T32, T33
    print('\t[Topic Properties] ---> %d/%s' % (series_index, topics_series))
    D, T1, T2, T31, T32, T33 = properties_for_topic(all_topics=topics_dict, topic_tracklet=topics_series)

    # 3.2) Topic Sponsor: P11, P12, P131, P132, P133, P14(seconds)
    print('\t[Topic Sponsor] ---> %d/%s' % (series_index, topics_series))
    P11, P12, P131, P132, P133, P14 = influence_of_sponsor(all_topics=topics_dict, topic_tracklet=topics_series,
                                                           user_info=usr_info)

    # 3.3) Topic Opinion Leader: P21, P22
    print('\t[Topic Opinion Leader] ---> %d/%s' % (series_index, topics_series))
    P21, P22 = influence_of_opinion_leader(all_topics=topics_dict, topic_tracklet=topics_series, user_info=usr_info)

    # 3.4) Topic Participator: P31, P321, P322, P323, P33
    print('\t[Topic Participator] ---> %d/%s' % (series_index, topics_series))
    P31, P321, P322, P323, P33 = influence_of_participator(all_topics=topics_dict, topic_tracklet=topics_series,
                                                           user_info=usr_info)
    writer.writerow([series_index,
                     D, T1, T2, T31, T32, T33,
                     P11, P12, P131, P132, P133, P14,
                     P21, P22,
                     P31, P321, P322, P323, P33,
                     N11, N12])

save_file.close()

