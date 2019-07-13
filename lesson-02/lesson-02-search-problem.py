from bs4 import BeautifulSoup
from urllib.request import urlopen
from urllib.error import  HTTPError
import collections
from collections import defaultdict
import os

def get_metro_line(lists=[]):
    """
    获取一条地铁线路的信息，包括线路的名称和URL
    :param lists: 输入的是TR对象，每个线路都有两行TR，用来表示起止地址
    :return: 线路名称，线路URL
    """
    tr = lists[0]
    td_tags = tr.children
    tds = list(td_tags)
    href = tds[0].next_element.attrs['href']  # 北京地铁1号线
    line = tds[0].text                        # 北京地铁1号线
    s1 = tds[1].text.replace('\n', ' ')        # 苹果园
    tr = lists[1]
    td_tags = tr.children
    tds = list(td_tags)
    s2 = tds[0].text.replace('\n', ' ')         # 四惠东
    print('line :{} , station :{} - {} , href :{}'.format(line, s1, s2, href))
    return (line, href)

def init_line(url = ''):
    """
    从百科北京地铁的首页，开始爬取全部的线路信息。然后再根据每个具体是线路，获取该线路的站点信息。
    :param url: 北京地铁的百科首页地址，有默认值，调用时可以不传。
    :return: 返回一个List[线路名称, 线路URL]
    """
    result = []
    try:
        if not url :
            url = 'https://baike.baidu.com/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%81/408485'
        page = urlopen(url)
        bsObj = BeautifulSoup(page, "html.parser")
        tables = bsObj.findAll("table", attrs={'log-set-param':'table_view', 'data-sort':'sortDisabled'} )
        table_tag = None
        for x in tables:
            tab_txt = x.text
            if tab_txt.find('起止点') >= 0:
                table_tag = x
                break
        tr_tags = table_tag.children
        tr_lists = list(tr_tags)
        tr_len = len(tr_lists)
        i = -1
        while i < tr_len-1:
            i += 1
            tr = tr_lists[i]
            if i == 0 : continue  # 跳过表头
            rowspan = int(tr.next_element.attrs.get('rowspan', '1'))
            if rowspan > 1 :      # 这是一条线路的数据，需要一起处理
                # print("tr_len = {} , i = {} , rowspan = {}".format(tr_len, i, rowspan))
                result.append(get_metro_line(tr_lists[i:i+rowspan]))
                i += rowspan - 1

    except HTTPError as e:
        print(e)
    return result

def get_line_detail(line_name ='', url = ''):
    """
    获取一个具体的地铁线路的站点信息。
    目前改函数没有对备用站点，未启用站点进行处理，导致实际结果会有更多的捷径可以走:（
    如果不对站点数据进行有效性的分析，使用正则处理会更加简单清晰，这个函数分析了table的
    DOM结构，程序实现上显得略微复杂
    :param line_name:线路的名称
    :param url:线路的百科URL
    :return:List[站点名称]
    """
    result = []
    # unique_station = set()
    try:
        if not url:
            url = 'https://baike.baidu.com/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%811%E5%8F%B7%E7%BA%BF'
        page = urlopen(url)
        bsObj = BeautifulSoup(page, "html.parser")
        tables = bsObj.findAll("table", attrs={'log-set-param':'table_view',
                                               'data-sort':'sortDisabled'} )
        table_tag = None
        for x in tables:
            tab_txt = x.text
            if tab_txt.find('换乘线路') >= 0:
                table_tag = x
                break

        tr_tags = table_tag.children
        tr_lists = list(tr_tags)
        tr_len = len(tr_lists)
        i = 1                    # 跳过标题, 表头
        while i < tr_len-1:
            i += 1
            tr = tr_lists[i]
            trs_list = list(tr.children)
            if not trs_list:
                continue
            # 获取TR下面的第一个TD，然后获取<A></A>的内容
            a_tag = get_a_label(trs_list[0])
            if a_tag:
                string_name = a_tag.string
                if string_name:
                    result.append(string_name)
                    print('线路 = {}, 车站 = {} , href = {}'.format(line_name, string_name, a_tag.attrs.get('href', '')))

    except HTTPError as e:
        print(e)
    return result


def get_a_label(tag_obj = None):
    """
    对给定的一个dom对象，判断是否为<a></a>标签
    :param tag_obj: DOM对象
    :return: 类型是<a></a>标签的DOM对象，或者是None
    """
    if tag_obj == None: return None
    obj = tag_obj.next

    name = obj.name
    if name == 'a':
        return obj
    elif type(obj).__name__ == 'Tag':
        return get_a_label(obj)
    else:
        return None


def is_goal(desitination):
    """
    判断是否是要到达的目标站点，这里的_wrap写法，需要再研究。
    :param desitination:目前站点的名称
    :return:
    """
    def _wrap(current_path):
        return current_path[-1] == desitination
    return _wrap


def search(graph, graph_dict, start, is_goal, search_strategy):
    """
    实现地铁上两个站点的导航搜索，目前支持了换乘最少和路径最短两种策略。
    :param graph: 结构为{key:[list]}的导航地图。本版本地图仅仅构建了一个方向，有时间应该构建两个方向的地图。
                  key=站点名称，list=该站点能够到达的下一个相邻站点
    :param graph_dict:结构为{key:{set}},key=地铁线路名称，set=该线路下的全部站点名称
    :param start:出发站点
    :param is_goal:到达站点
    :param search_strategy:搜索策略，目前实现了换乘最少和路径最短两种策略
           stratety_shortest_path , stratety_minimum_transfer
    :return:[站点名称] 计算出来的站点名称
    """
    pathes = [[start]]
    seen = set()

    while pathes:
        path = pathes.pop(0)
        froniter = path[-1]

        if froniter in seen: continue

        successors = graph[froniter]

        for city in successors:
            if city in path: continue

            new_path = path + [city]

            pathes.append(new_path)

            if is_goal(new_path): return new_path
        #        print('len(pathes)={}'.format(pathes))
        seen.add(froniter)

        pathes = search_strategy(pathes, graph_dict)


def save_stations(stations_con, station_dict):
    """
    将从百科上爬取的线路、站点信息保存到本地文件，方便调试运行。
    :param stations_con:  {key:[list]} , key=站点名称，list=该站点能够到达的下一个相邻站点
    :param station_dict:  {key:{set}}  , key=地铁线路名称，set=该线路下的全部站点名称
    :return:
    """
    if not stations_con: return
    file = open("stations.txt",'w')
    for x in stations_con:
        if not x: continue
        s = x + '->'
        s += ','.join(s for s in stations_con[x])
        file.write(s+'\n')
    file.close()

    file = open("stations_dict.txt", 'w')
    for x in station_dict:
        if not x: continue
        s = x + '->'
        s += ','.join(station_dict[x])
        file.write(s+'\n')
    file.close()


def load_stations(file_con, file_dict):
    """
    加载本地保存的站点信息
    :param file_con:
    :param file_dict:
    :return:
    """
    stations_con = defaultdict(list)
    stations_dict = defaultdict(set)
    file = open(file_con, 'r')
    lines = file.readlines()
    for line in lines:
        line = line.replace('\n', '')
        s = line.split('->')
        s[1].split(',')
        stations_con[s[0]] = s[1].split(',')
    file.close()

    file = open(file_dict, 'r')
    lines = file.readlines()
    for line in lines:
        line = line.replace('\n', '')
        s = line.split('->')
        s[1].split(',')
        for x in s[1].split(','):
            stations_dict[s[0]].add(x)
    file.close()

    return stations_con, stations_dict


def stratety_shortest_path(pathes, graph_dict):
    """
    用站点的数量来替代实际的距离，站点数量最少的，排序在最前
    :param pathes:待排序的已发现线路
    :return: 按线路节点数量进行排序
    """
    return sorted(pathes, key=len, reverse=False)


def stratety_minimum_transfer(pathes, graph_dict):
    """
    计算换乘线路的数量，得出最少换乘的排序在最前
    :param pathes:
    :param graph_dict: {key:{set}}  , key=地铁线路名称，set=该线路下的全部站点名称
    :return:
    """
    def get_transfer(path):
        # sorted函数默认的参数是待排序list的元素
        last_single_station = ''
        i_transfer = 0
        for x in path:
            temp_station = ''
            line = get_lines_by_station(graph_dict, x)
            if len(line) == 1: temp_station = line[0]

            if len(line) > 1: continue

            if last_single_station == '' and temp_station:
                last_single_station = temp_station
                continue

            if last_single_station != temp_station:
                last_single_station = temp_station
                i_transfer += 1
        return i_transfer

    return sorted(pathes, key=get_transfer, reverse=False)


def get_lines_by_station(stations_dic, s):
    """
    计算给定的地铁站点，对应的地铁线路。在换乘优先的推荐中，作为是否换乘的依据。
    :param stations_dic: {key:{set}}  , key=地铁线路名称，set=该线路下的全部站点名称
    :param s: 需要计算的站点名称
    :return: [list] 需要计算的站点所对应的地铁线路列表
    """
    result = []
    for x in stations_dic:
        ss = stations_dic[x]
        if s in ss:
            result.append(x)
    return result

def strarety_comprehensive():
    pass


if __name__ == '__main__':

    host = 'https://baike.baidu.com'
    station_connection = defaultdict(list)
    station_dict = defaultdict(set)

    # get_line_detail('北京地铁昌平线', host+'/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%81%E6%98%8C%E5%B9%B3%E7%BA%BF')
    # exit("end!")       # for debug

    # 初始化站点地图和站点字典
    if os.path.exists('stations.txt'):
        station_connection, station_dict = load_stations('stations.txt','stations_dict.txt')
    else:
        metro_lines = init_line(host + '/item/%E5%8C%97%E4%BA%AC%E5%9C%B0%E9%93%81/408485')
        ab_dict = defaultdict(set)
        ba_dict = defaultdict(set)
        for line, href in metro_lines:
            stations = get_line_detail(line, host+href)
            length = len(stations)
            for i in range(length-1):
                station_dict[line].add(stations[i + 1])
                print(' stations[i] = {} ,stations[i + 1]={}'.format(stations[i], stations[i + 1]))
                if not (stations[i + 1] in ab_dict[stations[i]]):
                    station_connection[stations[i]].append(stations[i + 1])
                ab_dict[stations[i]] = stations[i + 1]

                if not (stations[i] in ba_dict[stations[i + 1]]):
                    station_connection[stations[i + 1]].append(stations[i])
                ba_dict[stations[i]] = stations[i + 1]

        if station_connection:
            save_stations(station_connection, station_dict)

    # 进行路径计算,最短路径
    # stations = search(station_connection,
    #                   station_dict,
    #                   start = '苹果园站',
    #                   is_goal = is_goal('石门站'),
    #                   search_strategy = stratety_shortest_path)

    # # 进行路径计算,最少换乘
    stations = search(station_connection,
                      station_dict,
                      start = '苹果园站',
                      is_goal = is_goal('石门站'),
                      search_strategy = stratety_minimum_transfer)

    # 将结果进行输出，同时计算换乘信息，此处代码应当再封装一下。
    last_single_station = ''
    i_index = 0
    i_transfer = 0
    for x in stations:
        i_index += 1
        temp_station = ''
        line = get_lines_by_station(station_dict, x)
        if len(line) == 1:
            temp_station = line[0]

        if len(line) > 1:
            print('{}  站名：{} , 线路：{} , 换乘: {}'.format(i_index, x, line, i_transfer))
            continue

        if last_single_station == '' and temp_station:
            last_single_station = temp_station
            continue

        if last_single_station != temp_station:
            last_single_station = temp_station
            i_transfer += 1

        print('{}  站名：{} , 线路：{} , 换乘: {}'.format(i_index, x, line, i_transfer))