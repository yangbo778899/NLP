import random
import re
import jieba
import os
from collections import Counter


import pandas as pd
from functools import reduce
from operator import add, mul


human = """
human = 自己 寻找 活动
自己 = 我 | 俺 | 我们 
寻找 = 看看 | 找找 | 想找点
活动 = 乐子 | 玩的
"""

host = """
host = 寒暄 报数 询问 业务相关 结尾 
报数 = 我是 数字 号 ,
数字 = 单个数字 | 数字 单个数字 
单个数字 = 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 
寒暄 = 称谓 打招呼 | 打招呼
称谓 = 人称 ,
人称 = 先生 | 女士 | 小朋友
打招呼 = 你好 | 您好 
询问 = 请问你要 | 您需要
业务相关 = 玩玩 具体业务
玩玩 = 耍一耍 | 玩一玩
具体业务 = 喝酒 | 打牌 | 打猎 | 赌博
结尾 = 吗？"""


sales = """
sales = 找车 | 问车 | 买车
找车 = 称谓 打招呼 , 您对 汽车型号 的车感兴趣吗 ？
称谓 = 先生 | 女士
打招呼 = 您好 | 你好
汽车型号 = 雷克萨斯ES | 奔驰GLC | 宝马5系
问车 = 我想看下 汽车型号 的车有没有 参配 ？
参配 = 单个参配 | 参配 单个参配
单个参配 = 全景天窗、| 主动刹车、 | 真皮座椅、 | 隐私玻璃、 | 自动大灯、 | 无线充电、
买车 = 打招呼 ,  汽车型号 带有 单个参配 的车多少钱？
"""


def create_human_grammar(grammar_str, split='=', line_split='\n'):
    grammar = {}
    for line in grammar_str.split(line_split):
        if not line.strip():
            continue
        exp, stmt = line.split(split)
        grammar[exp.strip()] = [s.split() for s in stmt.split('|')]
    return grammar


choice = random.choice


def generate(gram, target):
    if target not in gram: return target # means target is a terminal expression
    
    expaned = [generate(gram, t) for t in choice(gram[target])]
    x = ''.join([e if e != '/n' else '\n' for e in expaned if e != 'null'])
    return x


def generate_n(num):
    for i in range(num):
        gramer = create_human_grammar(sales)
        x = generate(gram=gramer, target='sales')
        i_pos = x.rfind('、')
        if i_pos > 0:
            x = x[:i_pos] + x[i_pos + 1:]
        print(x)

# generate_n(9)

def token(string):
    # we will learn the regular expression next course.
    return re.findall('\w+', string)

def cut(string):
    return list(jieba.cut(string))


def train_token():
    global BASE_HOME
    # filename = BASE_HOME + 'train.txt'
    # filename_clean = BASE_HOME + 'train_clean.txt'

    filename = BASE_HOME + 'movie_comments.csv'
    filename_clean = BASE_HOME + 'movie_comments_clean.txt'

    articles = []
    global TOKEN
    global WORDS_COUNT

    for i, line in enumerate((open(filename))):
        if i % 1000 == 0: print(i)
        # replace 10000 with a big number when you do your homework.
        # x = line.split('++$++')
        x = line.split(',')
        if len(x) > 3 : articles.append(x[3])

    # with_jieba_cut = Counter(jieba.cut(articles[110]))

    articles_clean = [''.join(token(str(a)))for a in articles]

    with open(filename_clean, 'w') as f:
        for a in articles_clean:
            f.write(a + '\n')
        f.close()

    for i, line in enumerate((open(filename_clean))):
        if i % 1000 == 0: print(i)
        TOKEN += cut(line)


def load_token():
    global TOKEN
    global TOKEN_2_GRAM
    global WORDS_COUNT
    f_token = BASE_HOME + 'token.txt'
    f_token_2 = BASE_HOME + 'token_2.txt'
    if os.path.exists(f_token):
        for i, line in enumerate((open(f_token))):
            TOKEN.append(line)

    if os.path.exists(f_token_2):
        for i, line in enumerate((open(f_token_2))):
            TOKEN_2_GRAM.append(line)


def save_token():
    f_token = BASE_HOME + 'token.txt'
    f_token_2 = BASE_HOME + 'token_2.txt'
    with open(f_token, 'w') as f:
        for a in TOKEN:
            f.write(a + '\n')
        f.close()

    with open(f_token_2, 'w') as f2:
        for a in TOKEN_2_GRAM:
            f2.write(a + '\n')
        f2.close()

def prob_1(word):
    global TOKEN
    global WORDS_COUNT
    return WORDS_COUNT[word] / len(TOKEN)


def prob_2(word1, word2):
    global TOKEN_2_GRAM
    global WORDS_COUNT_2

    if word1 + word2 in WORDS_COUNT_2:
        return WORDS_COUNT_2[word1+word2] / len(TOKEN_2_GRAM)
    else:
        return 1 / len(TOKEN_2_GRAM)


def get_probablity(sentence):
    words = cut(sentence)
    sentence_pro = 1

    for i, word in enumerate(words[:-1]):
        next_ = words[i + 1]
        probability = prob_2(word, next_)
        sentence_pro *= probability
    return sentence_pro

def generate_best():
    gramer = create_human_grammar(sales)
    result = []
    for sen in [generate(gram=gramer, target='sales') for i in range(10)]:
        y = get_probablity(sen)
        result.append((sen, y))
        # sorted(result, key=lambda x: x[1], reverse=True)

        # print('sentence: {} with Prb: {}'.format(sen, get_probablity(sen)))
    arrx = sorted(result, key=lambda x: x[1], reverse=True)
    for e in arrx: print(e)
    print("the best :{}".format(arrx[0]))



# frequiences = [f for w, f in words_count.most_common(100)]
# x = [i for i in range(100)]
# import matplotlib.pyplot as plt
# plt.plot(x, frequiences)
# plt.show()

BASE_HOME = '/Users/yangbo/working/机器学习/深度学习/作业/lesson-01/'
TOKEN = []
TOKEN_2_GRAM = []
WORDS_COUNT = []
WORDS_COUNT_2 = []

load_token()
if (len(TOKEN) <= 0 and len(TOKEN_2_GRAM) <= 0):
    train_token()
    TOKEN = [str(t) for t in TOKEN]
    TOKEN_2_GRAM = [''.join(TOKEN[i:i + 2]) for i in range(len(TOKEN[:-2]))]
    save_token()

WORDS_COUNT = Counter(TOKEN)
WORDS_COUNT_2 = Counter(TOKEN_2_GRAM)

print(prob_1('我们'))
print(TOKEN[:10])
print(TOKEN_2_GRAM[:10])

print(prob_2('我们', '在'))
print('sentence: 小明今天抽奖抽到一台苹果手机 with Prb {}'.format(get_probablity('小明今天抽奖抽到一台苹果手机')))

generate_best()