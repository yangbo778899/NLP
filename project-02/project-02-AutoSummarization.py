from pyltp import Segmentor
from scipy.spatial.distance import cosine
from pyltp import SentenceSplitter
from gensim.models import FastText
from gensim.models.word2vec import LineSentence
from collections import Counter
from functools import partial
from sklearn.decomposition import PCA
import matplotlib.font_manager as fm
import networkx
import pandas as pd
import numpy as np
import os
import re

def cut(string):
    global segmentor
    words = segmentor.segment(string)  # 分词
    return ' '.join(words)



fp1 = fm.FontProperties(fname="/Users/mqgao/Downloads/SourceHanSerifSC-Regular.otf")


# how to set chinese display:
# https://knowlab.wordpress.com/2016/05/25/networkx-%E7%B9%AA%E5%9C%96%E9%A1%AF%E7%A4%BA%E4%B8%AD%E6%96%87%E7%9A%84%E8%A7%A3%E6%B1%BA%E6%96%B9%E6%B3%95/


def split_sentence(sentence):
    #     pattern = re.compile('[。；]')
    #     split = pattern.sub('\r', sentence).split()  # split sentence
    #     return split

    sents = SentenceSplitter.split(sentence)
    result = [x for x in sents if x != '']
    return result


def get_summarization_simple_with_text_rank(text, score_fn_type, constraint, window):
    return get_summarization_simple(text, score_fn_type, constraint, window)


def sentence_ranking_by_text_ranking(sentence, windows):
    '''
    根据构造好的图，计算pagerank
    '''
    sentence_graph = get_connect_graph_by_text_rank(sentence, windows)
    ranking_sentence = networkx.pagerank(sentence_graph)
    ranking_sentence = sorted(ranking_sentence.items(), key=lambda x: x[1], reverse=True)
    return ranking_sentence


def get_connect_graph_by_text_rank(split_sentence, window=3):
    '''
    将文章分句后，构造左右是windows数量的图
    '''
    keywords_graph = networkx.Graph()
    tokeners = split_sentence
    #     tokeners = tokenized_text
    for ii, t in enumerate(tokeners):
        word_tuples = [(tokeners[connect], t)
                       for connect in range(ii - window, ii + window + 1)
                       if connect >= 0 and connect < len(tokeners)]
        keywords_graph.add_edges_from(word_tuples)

    return keywords_graph


def get_summarization_simple(text, score_fn_type, constraint, windows=3):
    '''
    提取摘要内容，这里获取句向量的相似度有两种方式：
    1、根据句子构造的图，利用pagerank算出来 -- sentence_ranking_by_text_ranking
    2、根据句向量和全文向量的相似度 -- get_corrlations
    '''
    sub_sentence = split_sentence(text)
    #     sub_sentence = cut(text).split()

    if score_fn_type == 'networkx':
        ranking_sentence = sentence_ranking_by_text_ranking(sub_sentence, windows)
    else:
        ranking_sentence = get_corrlations(sub_sentence, cut)

    selected_text = set()
    current_text = ''

    for sen, _ in ranking_sentence:
        if len(current_text) < constraint:
            current_text += sen
            selected_text.add(sen)
        else:
            break

    summarized = []
    for sen in sub_sentence:  # print the selected sentence by sequent
        if sen in selected_text:
            summarized.append(sen)

    return summarized


def get_corrlations(text, cut_fn):
    """
    计算句子和全文的余弦相似度
    :param text: 返回
    :param cut_fn:
    :return:
    """

    if isinstance(text, list):
        text_ = ' '.join(text)
    else:
        text_ = text

    frequence = get_frequence()

    sub_sentences = split_sentence(text_)

    sentence_vector = sentence_embedding(text_, cut_fn, frequence)

    correlations = {}

    for sub_sentence in sub_sentences:
        sub_sen_vec = sentence_embedding(sub_sentence, cut_fn, frequence)
        correlation = cosine(sentence_vector, sub_sen_vec)
        correlations[sub_sentence] = correlation

    return sorted(correlations.items(), key=lambda x: x[1], reverse=True)


def sentence_embedding(sentence, cut, frequence):
    """
    利用词向量模型，生成句向量
    :param sentence:需要向量化的句子。
    :param cut:分词函数
    :param frequence:词频统计结果
    :return:生成的句向量
    """
    # weight = alpah/(alpah + p)
    # alpha is a parameter, 1e-3 ~ 1e-5
    alpha = 1e-3

    max_fre = max(frequence.values())

    words = cut(sentence).split()

    sentence_vec = np.zeros_like(model.wv.__getitem__('测试'))

    words = [w for w in words if model.wv.__contains__(w)]

    for w in words:
        weight = alpha / (alpha + frequence.get(w, max_fre))
        sentence_vec += weight * model.wv.__getitem__(w)

    sentence_vec /= len(words)
    # Skip the PCA
    #     estimator = PCA(n_components=1)
    #     sentence_vec_pca = estimator.fit_transform(sentence_vec.reshape(-1, 5))
    #     sentence_vec_pca = sentence_vec_pca.reshape(1,-1)

    return sentence_vec

def get_frequence():
    """
    Counter({'此外': 4242,
         '，': 1360791,
         '自': 6100,
         '本周': 520,
         '（': 212005,
         '6月': 17525,
         '12日': 4841,
         '）': 212463,
         '起': 13791,
    counter统计了tokeners.txt中，各个分词出现的次数。
    :return:
    {'此外': 0.0001890995140873566,
     '，': 0.060661225100058475,
     '自': 0.0002719252795692775,
     '本周': 2.3180515635413817e-05,
     '（': 0.009450740802472898,
     '6月': 0.0007812279548281292,
     '12日': 0.00021580168498276596,
     '）': 0.009471157487397935,
     '起': 0.0006147740213999845,
     返回各个分词在统计结果中的占比
        """
    tokeners = []
    if os.path.exists('./res/tokeners.txt'):
        with open('./res/tokeners.txt','r', encoding='gb18030') as tf:
            line = tf.readline()
            while line:
                tokeners.append(line)
                line = tf.readline()
    else:
        tokeners = [t for l in pure_content['tokenized_content'].tolist() for t in l.split()]
        with open('./res/tokeners.txt','a', encoding='gb18030') as tf:
            tf.write('\n'.join(tokeners))

    tokener_counter = Counter(tokeners)
    frequence = {w: count/len(tokeners) for w, count in tokener_counter.items()}

    # occurences_frequences = sorted(list(frequence.values()), reverse=True)
    # X = range(len(occurences_frequences))
    return frequence


if __name__=="__main__":

    path_root = '../data'
    news_file = os.path.join(path_root, 'sqlResult_1558435.csv')
    news_content = pd.read_csv(news_file, encoding='gb18030')
    pure_content = pd.DataFrame()
    pure_content['content'] = news_content['content']
    pure_content = pure_content.fillna('')

    LTP_DATA_DIR = '../project-01/res/ltp_data_v3.4.0/'  # ltp模型目录的路径
    # 加载分词模型
    segmentor = Segmentor()  # 初始化实例
    cws_model_path = os.path.join(LTP_DATA_DIR, 'cws.model')
    segmentor.load(cws_model_path)  # 加载模型

    # 对内容进行分词，并保存在 tokenized_content 列中，如果不加载之前跑好的结果，这里有点费时间。
    if os.path.exists('./res/pure_content_cut.csv'):
        pure_content = pd.read_csv('./res/pure_content_cut.csv', keep_default_na=False)
    else:
        pure_content['tokenized_content'] = pure_content['content'].apply(cut)
        pure_content.to_csv('./res/pure_content_cut.csv')

    if not os.path.exists('./res/pure_content_cut.csv'):
        with open('./res/all_corpus.txt', 'w', encoding='gb18030') as f:
            f.write(' '.join(pure_content['tokenized_content'].tolist()))


    # 加载FastText模型，如果模型没有的话，就训练一个。
    model_file = './res/fastText.model'
    if os.path.exists(model_file):
        model = FastText.load(model_file)
    else:
        model = FastText(LineSentence('all_corpus.txt'), window=5, size=35, iter=10, min_count=2)
        model.save('./res/fastText.model')


    # 开始跑摘要文章啦！！
    index = 2496
    constraint = 200
    document = pure_content.iloc[index]['content']

    s1 = ' '.join(get_summarization_simple_with_text_rank(document,
                                                    'networkx',
                                                    constraint,
                                                    window=1))

    s2 = ' '.join(get_summarization_simple(document,
                                      'cosin',
                                      constraint,
                                      windows=0))

    print('source:')
    for x in split_sentence(document):
        print(x)

    print('')

    print('networkx:')
    for x in split_sentence(s1):
        print(x)

    print('')

    print('cosin:')
    for x in split_sentence(s2):
        print(x)
'''
source:
原标题：中国墨子号卫星首次实现1200公里量子纠缠，震惊国外专家
雷锋网按：在量子加密通信的研究领域，如何长距离传输纠缠光子一直是个很大的难题。
不过最近我国的科学家们，利用去年八月发射的墨子号量子卫星，在这件事上取得了一些突破。
研究者们成功从太空中，往相距约 1200 公里的云南丽江和青海德令哈地面站发送了纠缠光子对。
本文由雷锋网编译。
去年年底的某个晴朗的夜晚，一个绿色的星点出现在中缅边界的地平线附近。
正在丽江郊外观测站的物理学家，中国科技大学教授陆朝阳观测到了这一现象，他说：“这很像一个非常明亮的绿色星体。”
陆教授和他的同事们必须赶快采取行动。
因为那个绿色星体其实是来自于 300 英里上空正在运行的卫星发出的一道激光，它就像一个灯塔预示着太空飞行器的位置。
激光点在空中快速移动着，10 分钟后就会消失在地平线中。
所以这个由中国的多个科学机构研究者组成的团队，正在用望远镜紧紧盯着这道绿光，努力捕捉着其中最重要的东西：这个卫星上由特殊晶体制造的一种微妙的单一红外光子。
最终他们过滤掉绿光，锁定了他们的“猎物”——一个过去从未发出射过的量子信号。
这项实验是量子密码这种新技术中的一次关键测试。
量子密码就是像光子这样的量子传输安全信息的技术。
但是众所周知，脆弱的量子不易进行传递，如果你试图利用光纤来传递它们，超过 150 英里后，信号就会失效，这种性质使得量子密码在全国或者世界范围内传递消息时起不到什么作用。
所以科学家们一直在研究如何利用卫星来进行长距离的量子传送。
但是在此之前，还没有人做到如此远的距离。
在这项实验中，中国科学家在两个相距 750 英里的地面站点和卫星之间传递单一光子，创造了距离最远的新记录（两站分别是青海德令哈站和云南丽江高美古站，两站距离1203公里）。
参与这项工作的巴黎量子计算中心副主管 Eleni Diamanti 说：“这个实验对扩展远距离量子沟通网络来讲拥有里程碑式的意义，这毫无疑问。”
去年 8 月，在戈壁滩的酒泉卫星发射中心，中国发射了造价一亿美元的量子卫星“墨子号”，专门用来进行空间级别的量子实验。
该卫星上搭载了一套复杂的激光系统、反射镜面系统和一中特殊的晶体，当激光反射在晶体上时，它会创造出一对处于纠缠态的光子。
晶体一次可以制造 6 百万对光子，但是地面上的两个站点每秒只能探测到大约一对光子。
陆教授说：“这项任务非常具有挑战性，类似于你在 300 米外观察一根头发。”
陆教授和他的同事们认为，量子密码技术会成为未来一种良好的加密工具，其工作原理是这样的：首先，测量光子的性质，得到一串由 0、1 组成的密钥，接着利用这串密钥加密你的信息并将其发送给特定的接收者。
如果黑客想要在传输中窃取这串密钥，根据测不准理论，量子将会在窃取的瞬间改变密钥数字。
想象薛定谔的猫，当你没有观察它时，它既生又死，而你一旦观察它，它就会表现出生或者死的一个状态。
同样的，偷窃的黑客会瞬间改变构成密钥的光子的状态，所以理论上，在理想状态下，这串密钥绝对不会被窃取（现实中的硬件设备并不完美，探测器在探测连续单一光子时表现不佳，这可能让我们误以为信息被窃取了，黑客也可以通过发射强光来追踪你的探测器）。
中国的量子卫星发射和这项实验是科研人员长期努力的结果。
负责这个项目的物理学家，中国科技大学教授潘建伟说，卫星实验的开始可以追溯到2003年，他带领的大约 100 人的团队从设计、建造到调整激光和卫星系统付出了多年的努力。
他们最初的实验是在地面上进行的，起初只是在几英里内传输密钥，后来慢慢开始加大距离。
“但是他们在该领域的研究仍然是很快的。”
加拿大滑铁卢大学的物理学 Thomas Jennewein 说道（他最近完成了从地面到飞行中的飞机间的量子传输）。
几年前，Jennewein 在国际空间站上试图完成相似的实验。
他说：“因为各种实验的复杂性、高昂的成本等等，那些项目没有一个可以实现如此远的距离。
但是中国团队现在做到了，他们走在了领域的前沿，这非常棒。”
杜克大学的中国科技政策研究专家 Denis Simon 说：“他们之所以行动如此快速，得益于中国政府对该项目的充分重视。
因为中国政府领导希望完成这样的实验，所以实验团队无需在通常的官僚制度上浪费时间。”
中国政府对量子通信技术抱有极大的兴趣，因为量子安全通信对国家利益大有裨益。
他说道：“中国政府想将这种通信技术运用在中国南海的海军战舰上，该技术的应用还有很多。”
同时，其他国家的科学家也在进行类似的实验，但是却被很多官僚制度所束缚。
比如 Diamanti 的团队还正在等待欧洲空间局对他们在国际空间站和欧洲几个地面站点间传递量子实验申请的回复。
伊利诺伊香槟分校的物理学家 Paul Kwiat 也正在领导美国的团队与 NASA 合作进行相似的实验。
但是还没有一个国家像中国一样对量子通信有着宏伟的计划。
陆教授和他的团队正在计划在一个新的更远的卫星上实施同样的实验，将量子通信的距离从城市间扩展得到更远。
他们想要在中国和奥地利（那里有一些合作伙伴）之间交换量子密钥。
潘建伟曾说过，中国计划在 2030 年打造一个覆盖全球的量子卫星通信网络。
陆教授说：“我们是非常幸运的，我们的成功得益于中国政府的快速决策系统，政治和科学的结合可以事半功倍。”

networkx:
雷锋网按：在量子加密通信的研究领域，如何长距离传输纠缠光子一直是个很大的难题。
不过最近我国的科学家们，利用去年八月发射的墨子号量子卫星，在这件事上取得了一些突破。
研究者们成功从太空中，往相距约 1200 公里的云南丽江和青海德令哈地面站发送了纠缠光子对。
陆教授和他的团队正在计划在一个新的更远的卫星上实施同样的实验，将量子通信的距离从城市间扩展得到更远。
他们想要在中国和奥地利（那里有一些合作伙伴）之间交换量子密钥。
潘建伟曾说过，中国计划在 2030 年打造一个覆盖全球的量子卫星通信网络。

cosin:
研究者们成功从太空中，往相距约 1200 公里的云南丽江和青海德令哈地面站发送了纠缠光子对。
本文由雷锋网编译。
激光点在空中快速移动着，10 分钟后就会消失在地平线中。
这项实验是量子密码这种新技术中的一次关键测试。
想象薛定谔的猫，当你没有观察它时，它既生又死，而你一旦观察它，它就会表现出生或者死的一个状态。
他们想要在中国和奥地利（那里有一些合作伙伴）之间交换量子密钥。
潘建伟曾说过，中国计划在 2030 年打造一个覆盖全球的量子卫星通信网络。
'''