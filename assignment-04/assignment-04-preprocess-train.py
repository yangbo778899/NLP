from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import gensim
import jieba
import os
import time
import re
import pandas as pd
from hanziconv import HanziConv
pd.options.mode.chained_assignment = None

import numpy as np
import nltk
import sys
from gensim.models import word2vec

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# stop_words_filename = './extracted/AA/stop_words.txt'
# term_filename = './extracted/AA/sentennce_cut.txt'
# wiki_filenames = ['./extracted/AA/wiki_00', './extracted/AA/wiki_01', './extracted/AA/wiki_02']
# wiki_chn_filenames = ['./extracted/AA/wiki_00_chn', './extracted/AA/wiki_01_chn', './extracted/AA/wiki_02_chn']
# model_filename = './extracted/AA/word2vec.model'
# pic_filename = './extracted/AA/modle_term.jpg'

stop_words_filename = './res/stop_words.txt'
term_filename = './res/sentennce_cut.txt'
wiki_filenames = ['./res/wiki_00', './res/wiki_01', './res/wiki_02']
wiki_chn_filenames = ['./res/wiki_00_chn', './res/wiki_01_chn', './res/wiki_02_chn']
model_filename = 'word2vec.model'
pic_filename = './res/modle_term.jpg'


PUNCTUATION_PATTERN = r'\”|\《|\。|\{|\！|？|｡|\＂|＃|＄|％|\＆|\＇|（|）|＊|＋|，|－|／|：|；|＜|＝|＞|＠|\［|\＼|\］|\＾|＿|｀|\～|｟|｠|\、|〃|》|「|」|『|』|【|】|〔|〕|〖|〗|〘|〙|〚|〛|〜|\〝|\〞|〟|〰|〾|〿|–—|\‘|\“|\„|\‟|\…|\‧|﹏|\.'

def punctuation(ustring):
    '''
    去掉正文内容中的非文字字符
    :param ustring: 文本内容
    :return: 处理后的文本
    '''
    return re.sub(PUNCTUATION_PATTERN, '', ustring)

def load_stop_words(name):
    '''
    加载停用词
    :param name:停用词的本地存放文件
    :return: 停用词集合set()
    '''
    words = set()
    file = open(name, 'r', encoding="utf-8")
    for word in file.readlines():
        words.add(word.strip())
    file.close()
    return words

def tran_wiki_chn():
    '''
    将wiki解压出来的文件，转化为简体中文。
    :return:
    '''

    for filename , filename_chn in zip(wiki_filenames, wiki_chn_filenames):
        f_wiki = open(filename, 'r', encoding='utf-8')
        f_chn = open(filename_chn, 'w', encoding='utf-8')
        i_count = 0
        while True:
            line = f_wiki.readline()
            if line == '':
                break
            simplified_sentence = HanziConv.toSimplified(line)
            f_chn.write(simplified_sentence + '\n')
            i_count += 1
            if i_count % 1000 == 0:
                print(i_count)
        f_wiki.close()
        f_chn.close()

def gen_terms():
    '''
    将转化成简体汉字的wiki文本，处理成分词，保存到定义好的文件中。
    :return:
    '''
    STOP_WORDS = load_stop_words(stop_words_filename)
    term_file = open(term_filename, 'w', encoding="utf-8")

    i_count = 0
    for name in wiki_chn_filenames:
        wiki_file = open(name, 'r', encoding="utf-8")
        content_line = wiki_file.readline()
        # 定义一个字符串变量，表示一篇文章的分词结果
        article_contents = ""
        while content_line:
            content_line = content_line.strip()
            if len(content_line) == 0:
                # 跳过空行
                content_line = wiki_file.readline()
                i_count += 1
                continue

            if content_line.find('<doc') >= 0:
                # 跳过文章开头
                article_contents = ""
                content_line = wiki_file.readline()
                i_count += 1
                continue

            if content_line.find('</doc>') >= 0:
                # 保存一篇文章
                term_file.write(article_contents + "\n")
                article_contents = ""

            # 处理非文本字符
            content_line = punctuation(content_line)
            if len(content_line) > 0:
                words = jieba.cut(content_line, cut_all=False)
                for word in words:
                    if word not in STOP_WORDS:
                        article_contents += word + " "
            content_line = wiki_file.readline()
            i_count += 1
            if i_count % 100 == 0:
                print(i_count)
        wiki_file.close()

    term_file.close()


def tsne_plot(model):
    '''
    Creates and TSNE model and plots it
    :param model: 训练好的词向量模型
    :return:
    '''

    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(pic_filename)
    # plt.show()

if __name__=="__main__":

    t_start = time.process_time()
    if os.path.exists(model_filename):
        model = gensim.models.Word2Vec.load(model_filename)
    else:
        gen_terms()

        common_texts = LineSentence(term_filename)

        path = get_tmpfile(model_filename)

        model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=2)

        model.save(model_filename)

    t_end = time.process_time()

    print( 'time:', t_end - t_start, model.wv.most_similar('建议', topn=10))

    t_end = time.process_time()

    print('success get most similar, time = {}'.format(t_end - t_start))
    # 这里执行太耗时了，没有能够跑出结果
    tsne_plot(model)
    t_end = time.process_time()

    print('success tsne_plot, time = {}'.format(t_end - t_start))