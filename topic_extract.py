# -*- coding: utf-8 -*-
# @Time    : 2023/11/6 下午 4:09
# @Author  : Guoshuai Zhang
# @E-mail  : guoshuai0914(a)126.com
# @File    : topic_extract.py
# @Software: PyCharm
# Copyright(c) 2023, Guoshuai Zhang. All Rights Reserved.


from bertopic import BERTopic
from datetime import datetime, timedelta
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pandas as pd
import pickle


def remove_emoji(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U00010000-\U0010FFFF"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


def process_articles(text):
    text = text.lower()
    text = text.replace('【', '')
    text = text.replace('】', '')
    text = text.replace('《', '')
    text = text.replace('》', '')
    text = re.sub('[\u4e00-\u9fa5]', ' ', text)  # chinese
    text = re.sub(r'/', ' ', text)
    text = re.sub(r'//', ' ', text)
    text = re.sub(r'\\', ' ', text)
    text = re.sub(r'\\\\', ' ', text)
    text = re.sub(r'-', ' ', text)
    text = re.sub(r'--', ' ', text)
    text = re.sub(r'—', ' ', text)
    text = re.sub(r'\d+', '', text)
    words = word_tokenize(remove_emoji(text=text))
    english_stopwords = stopwords.words('english')
    addition_stopwords = ['\'re', '\'s', '\'t', '\'m', '\'ll', '\'ve', '\'d', 'n\'t']
    english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '-',
                            '..', '...', '......', '|', '``', '\'', '\'\'', '<', '>', '~', '+', '/', '//', '"']
    nonsense_words = ['thanks', 'thank', 'yes', 'good', 'one', 'hello', 'lol', 'nan', 'ok', 'well', 'hi', 'please',
                      'go', 'get', 'see', 'going', 'two', 'better', 'us', 'yeah', 'sorry', 'much', 'red', 'no', 'like',
                      'everyone', 'let', 'already', 'many', 'people', 'also', 'know', 'us', 'okay', 'four', 'three',
                      'name', 'time', 'minutes', 'close', 'take', 'finish', 'need', 'come', 'done', 'day', 'way',
                      'still', 'got', 'work', 'wait', 'per', 'person', 'welcome', 'min', 'minute', 'seconds', 'second',
                      'haha', 'ah', 'mm', 'eye', 'whatever', 'lisa', 'yana', 'alice', 'got', 'always', 'tin', 'may', 'pie',
                      'rc', 'ra', 'per', 'pls', 'thx', 'could', 'oh', 'ja', 'should', 'acs', 'xd', 'think', 'sho',
                      'morning', 'utc', 'last', 'halk', 'axel', 'afk', 'hmm', 'ha', 'argo', 'rdm', 'senti', 'elysium',
                      'eon', 'hyeon', 'elypsium', 'nvm', 'nzm', 'cobos', 'snd', 'aether', 'truce', 'razorman', 'ooo',
                      'loool', 'hell', 'karen', 'muxayo', 'murmansk', '|||', '++', 'plz', 'v', 'soon', 'yet', 'np', 'ok.',
                      'hz', 'k', 'les']

    filter_words = [w for w in words if (w not in english_stopwords)]
    filter_words = [w for w in filter_words if (w not in addition_stopwords)]
    filter_words = [w for w in filter_words if (w not in english_punctuations)]
    filter_words = [w for w in filter_words if (w not in nonsense_words)]

    return ' '.join(filter_words)


file_ = '20230713_1207_19741953'
content_dict = dict()
topics_dic = dict()

# 1) Read and process
en_conversation = pd.read_csv('./%s_en.tsv' % file_, sep='\t', encoding='utf-8')
for history_row in en_conversation.iterrows():
    split_line = history_row[-1]
    time_stamp = split_line[2][:10]
    content = str(split_line[5]).strip()
    pro_content = process_articles(content)

    if time_stamp in list(content_dict.keys()):
        if pro_content:
            content_dict[time_stamp]['pro_content'].append(pro_content)
            content_dict[time_stamp]['content'].append(content)
            content_dict[time_stamp]['time_stamp'].append(split_line[2])
            content_dict[time_stamp]['uid'].append(str(split_line[0]).strip())
        else:
            continue
    else:
        content_dict[time_stamp] = {'content': list(), 'pro_content': list(), 'time_stamp': list(), 'uid': list()}
        if pro_content:
            content_dict[time_stamp]['pro_content'].append(pro_content)
            content_dict[time_stamp]['content'].append(content)
            content_dict[time_stamp]['time_stamp'].append(split_line[2])
            content_dict[time_stamp]['uid'].append(str(split_line[0]).strip())
        else:
            continue

# 2) Extracting topics
for time_key in list(content_dict.keys()):

    try:
        print(datetime.strftime(datetime.utcnow(), '%Y-%m-%d %H:%M:%S'),
              'extracting topics for Date: %s ...' % time_key)

        topic_model = BERTopic(nr_topics='auto')
        topics, probs = topic_model.fit_transform(content_dict[time_key]['pro_content'])

        topic_docs = {topic: list() for topic in set(topics)}
        for topic, doc in zip(topics, content_dict[time_key]['content']):
            topic_docs[topic].append(doc)

        # 3) Save Topics
        latest_topics = dict()
        for topic_iter, topic_info in enumerate(topic_model.get_topic_info().iterrows()):
            topic_name = topic_info[-1][0]
            topic_count = topic_info[-1][1]
            topic_keywords = topic_info[-1][3]
            topic_docs_index = [content_dict[time_key]['content'].index(d) for d in topic_docs[topic_name]]

            topic_embedding = topic_model.topic_embeddings_[topic_iter, :]
            if topic_name != -1:
                latest_topics[topic_name] = {'topic_words': topic_keywords, 'count': topic_count,
                                             'topic_embedding': topic_embedding,
                                             'topic_content': topic_docs[topic_name],
                                             'topic_ts': [content_dict[time_key]['time_stamp'][i_]
                                                          for i_ in topic_docs_index],
                                             'topic_uid': [content_dict[time_key]['uid'][i_]
                                                           for i_ in topic_docs_index]
                                             }
            else:
                continue

        topics_dic[time_key] = latest_topics

    except BaseException as err:
        print(datetime.strftime(datetime.utcnow(), '%Y-%m-%d %H:%M:%S'),
              'BERTopic ERROR: Date: %s' % time_key, err)
        topics_dic[time_key] = dict()

# 4) Save into .pkl
with open('all_topics_%s.pkl' % file_, 'wb') as dic_f:
    pickle.dump(topics_dic, dic_f)
print(datetime.strftime(datetime.utcnow(), '%Y-%m-%d %H:%M:%S'), 'topics saved!\n')

# with open('my_dict.pickle', 'rb') as f:
#     my_dict = pickle.load(f)
