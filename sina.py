# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 21:35:36 2017

@author: dell
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 15:56:11 2017

@author: dell
"""

import os
import jieba

def TextProcessing(folder_path):
    folder_list = os.listdir(folder_path)
    data_list = []
    class_list = []
    
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)
        files = os.listdir(new_folder_path)
        
        j = 1
        for file in files:
            if i > 100:
                break
            with open(os.path.join(new_folder_path, file), 'r', encoding = 'utf-8') as f:
                raw = f.read()
            
            word_cut = jieba.cut(raw, cut_all = False)
            word_list = list(word_cut)
            
            data_list.append(word_list)
            class_list.append(folder)
            j += 1
        print(data_list)
        print(class_list)

if __name__ == '__main__':
    folder_path = 'SogouC/Sample'
    TextProcessing(folder_path)
    
#########################################
import os
import random
import jieba


def TextProcessing(folder_path, test_size = 0.2):
    folder_list = os.listdir(folder_path)                        #查看folder_path下的文件
    data_list = []                                                #数据集数据
    class_list = []                                                #数据集类别

    #遍历每个子文件夹
    for folder in folder_list:
        new_folder_path = os.path.join(folder_path, folder)        #根据子文件夹，生成新的路径
        files = os.listdir(new_folder_path)                        #存放子文件夹下的txt文件的列表

        j = 1
        #遍历每个txt文件
        for file in files:
            if j > 100:                                            #每类txt样本数最多100个
                break
            with open(os.path.join(new_folder_path, file), 'r', encoding = 'utf-8') as f:    #打开txt文件
                raw = f.read()

            word_cut = jieba.cut(raw, cut_all = False)            #精简模式，返回一个可迭代的generator
            word_list = list(word_cut)                            #generator转换为list

            data_list.append(word_list)                            #添加数据集数据
            class_list.append(folder)                            #添加数据集类别
            j += 1

    data_class_list = list(zip(data_list, class_list))            #zip压缩合并，将数据与标签对应压缩
    random.shuffle(data_class_list)                                #将data_class_list乱序
    index = int(len(data_class_list) * test_size) + 1            #训练集和测试集切分的索引值
    train_list = data_class_list[index:]                        #训练集
    test_list = data_class_list[:index]                            #测试集
    train_data_list, train_class_list = zip(*train_list)        #训练集解压缩
    test_data_list, test_class_list = zip(*test_list)            #测试集解压缩

    all_words_dict = {}                                            #统计训练集词频
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1

    #根据键的值倒序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key = lambda f:f[1], reverse = True)
    all_words_list, all_words_nums = zip(*all_words_tuple_list)    #解压缩
    all_words_list = list(all_words_list)                        #转换成列表
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list
        #print(data_list)
        #print(class_list)
if __name__ == '__main__':
    #文本预处理
    folder_path = 'SogouC/Sample'                #训练集存放地址
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)
    print(all_words_list)
    
def MakeWordsSet(words_file):
    words_set = set()
    with open(words_file, 'r', encoding = 'utf-8') as f:
        for line in f.readlines():
            word = line.strip()
            if len(word) > 0:
                words_set.add(word)
    return words_set

def words_dict(all_words_list, deleteN, stopwords_set = set()):
    feature_words = []                            #特征列表
    n = 1
    for t in range(deleteN, len(all_words_list), 1):
        if n > 1000:                            #feature_words的维度为1000
            break                               
        #如果这个词不是数字，并且不是指定的结束语，并且单词长度大于1小于5，那么这个词就可以作为特征词
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n += 1
    return feature_words

if __name__ == '__main__':
    #文本预处理
    folder_path = 'SogouC/Sample'                #训练集存放地址
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)

    #生成stopwords_set
    stopwords_file = 'stopwords_cn.txt'
    stopwords_set = MakeWordsSet(stopwords_file)

    feature_words = words_dict(all_words_list, 100, stopwords_set)
    print(feature_words)
    

    