"""
针对三个数据源的数据清洗代码

@Author: Cui Ruilong
@Time: 2019-02-27
"""

import random
import codecs
import os

import numpy as np


def load_sentences(path):
    """
    将原始语料转换成以句子为单位的列表
    :param path:
    :return:
    """
    sentences = []
    sentence = []
    for line in codecs.open(path, 'r', 'utf-8'):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                sentences.append(sentence)
            sentence = []
        else:
            word = line.split('\t')
            sentence.append(word)
    if len(sentence) > 0:
        sentences.append(sentence)
    return sentences


def get_data_label(label, name):
    """
    获取数据的标签，针对 data_2 的工具函数
    :param label:
    :param name:
    :return:
    """
    result = []
    name = name.strip().replace(" ", "")
    if label == 'O':
        for i in range(len(name)):
            result.append(name[i] + '\t' + 'O' + '\n')
    else:
        start = True
        for i in range(len(name)):
            if start:
                start = False
                result.append(name[i] + '\t' + 'B-' + label + '\n')
            else:
                result.append(name[i] + '\t' + 'I-' + label + '\n')

    return result


def divide_train_dev_test(save_dir, original_data_dir, train_data, dev_data=None, test_data=None):
    """
    将原始数据划分成 训练集：70% 验证集：10% 测试集：20%
    :param save_dir: 保存划分好的数据集的目录
    :param original_data_dir: 原始数据集目录
    :param train_data: 数据
    :param dev_data:
    :param test_data:
    :return:
    """
    np.random.seed(0)  # random seed
    pc = np.array([0.75, 0.1, 0.15])
    f_train = open(os.path.join(save_dir, 'train.txt'), 'w+', encoding='utf-8')
    f_test = open(os.path.join(save_dir, 'test.txt'), 'w+', encoding='utf-8')
    f_dev = open(os.path.join(save_dir, 'dev.txt'), 'w+', encoding='utf-8')
    train_num = 0
    test_num = 0
    dev_num = 0

    sentences = load_sentences(os.path.join(original_data_dir, train_data))

    if dev_data is not None:
        sentences.extend(load_sentences(os.path.join(original_data_dir, dev_data)))

    if test_data is not None:
        sentences.extend(load_sentences(os.path.join(original_data_dir, test_data)))

    random.shuffle(sentences)

    for sen in sentences:
        index = np.random.choice(['train', 'dev', 'test'], p=pc.ravel())
        if index == 'train':
            f_w = f_train
            train_num += 1
        elif index == 'test':
            f_w = f_test
            test_num += 1
        else:
            f_w = f_dev
            dev_num += 1

        for word in sen:
            f_w.write(word[0] + '\t' + word[1] + '\n')
        f_w.write('\n')

    f_train.close()
    f_test.close()
    f_dev.close()

    print('All sentences: {0}'.format(len(sentences)))
    print('Train sentences: {0}'.format(train_num))
    print('Test sentences: {0}'.format(test_num))
    print('Dev sentences: {0}'.format(dev_num))


def data_1_clear():
    """
    数据来源：https://github.com/crownpku/Small-Chinese-Corpus
    原始数据里面标签与字符之间是用空格隔开，现在替换成TAB符号隔开。并将新的数据保存到当前目录下
    训练集、测试集和验证集保持原来数据分布不变。
    产生三个文件：train.txt  test.txt  dev.txt
    :return:
    """

    def clear(save, read):
        with open(save, 'w+', encoding='utf-8') as f_s:
            for line in open(read, 'r', encoding='utf-8'):
                line = line.strip()
                if line == '':
                    f_s.write('\n')
                else:
                    line = line.split(' ')
                    f_s.write(line[0] + '\t' + line[1] + '\n')

    train_path = '../data/data_1/example.train'
    test_path = '../data/data_1/example.test'
    dev_path = '../data/data_1/example.dev'

    train_save = '../data/data_1/train.txt'
    test_save = '../data/data_1/test.txt'
    dev_save = '../data/data_1/dev.txt'

    print('Start process data_1...')
    clear(train_save, train_path)
    clear(test_save, test_path)
    clear(dev_save, dev_path)
    print('Done.')


def data_ori_clear():
    """
    数据来源：https://github.com/crownpku/Small-Chinese-Corpus
    原始数据里面标签与字符之间是用空格隔开，现在替换成TAB符号隔开。并将新的数据保存到当前目录下
    训练集、测试集和验证集保持原来数据分布不变。
    产生三个文件：train.txt  test.txt  dev.txt
    :return:
    """

    def clear(save, read):
        with open(save, 'w+', encoding='utf-8') as f_s:
            for line in open(read, 'r', encoding='utf-8'):
                line = line.strip()
                if line == '':
                    f_s.write('\n')
                else:
                    line = line.split(' ')
                    f_s.write(line[0] + '\t' + line[1] + '\n')

    train_path = './data/data_ori/example.train'
    test_path = './data/data_ori/example.test'
    dev_path = './data/data_ori/example.dev'

    train_save = './data/data_ori/train.txt'
    test_save = './data/data_ori/test.txt'
    dev_save = './data/data_ori/dev.txt'

    print('Start process data_ori...')
    clear(train_save, train_path)
    clear(test_save, test_path)
    clear(dev_save, dev_path)
    print('Done.')


def data_2_clear():
    """
    数据来源：https://bosonnlp.com/dev/resource
    将 BOSON命名实体识别数据 转换成 BIO 标签并保存到 all_data.txt 文件中
    并对 all_data.txt 文件里面的数据划分训练集、测试集和验证集
    原始数据标签如下：
        time: 时间
        location: 地点
        person_name: 人名
        org_name: 组织名
        company_name: 公司名
        product_name: 产品名
    产生四个文件：all_data.txt  train.txt  test.txt  dev.txt
    :return:
    """
    base_dir = '../data/data_2'
    raw_path = os.path.join(base_dir, 'BosonNLP_NER_6C.txt')
    save_path = os.path.join(base_dir, 'all_data.txt')
    dict = {'time': 'DATE', 'location': 'LOC', 'person_name': 'PER', 'org_name': 'ORG', 'company_name': 'ORG',
            'product_name': 'O'}
    segment = ['。', '!', '!', '?', '？']
    print('Start process data_2...')
    with open(save_path, 'w+', encoding='utf-8') as f_s:
        for line in open(raw_path, 'r', encoding='utf-8'):
            line = line.strip().replace(' ', '')
            line = line.replace('\\n\\n', '')
            line = line.replace('\t', '')
            line = line.replace('\xa0', '')
            index = 0
            length = len(line)
            sentences = []

            while True:
                if index >= length:
                    break

                if line[index] != "{":
                    sentences.append(line[index] + '\t' + 'O' + '\n')
                    index += 1
                elif line[index] == '{' and index + 2 < length and line[index + 1] == '{':
                    end_index = line[index:].find('}}')
                    if end_index == -1 or index + end_index >= length:
                        break

                    temp = line[index + 2: index + end_index]
                    if temp.find(":") == -1:
                        break

                    label_name = temp.split(":")
                    if len(label_name) != 2:
                        index = index + end_index + 2
                        continue
                    if label_name[0] in dict.keys():
                        result = get_data_label(dict[label_name[0]], label_name[1])
                        sentences.extend(result)
                    index = index + end_index + 2
                else:
                    break

            for word in sentences:
                f_s.write(word)
                if word[0] in segment:
                    f_s.write('\n')
            if len(sentences) != 0:
                last_word = sentences[-1]
                if last_word != "" and last_word[0] not in segment:
                    f_s.write('\n')
        divide_train_dev_test(base_dir, base_dir, 'all_data.txt')
        print('Done.')


def data_3_clear():
    """
    数据来源：https://github.com/Determined22/zh-NER-TF
    数据采用的是BIO格式的标签，无须再做转换处理
    原始数据格式为：train_data  和  test_data
    对原始数据划分训练集、测试集和验证集
    :return:
    """
    print('Start Process data_3...')
    base_dir = '../data/data_3'
    divide_train_dev_test(base_dir, base_dir, 'train_data', 'test_data')
    print('Done.')


def data_4_clear():
    """
    数据来源：从互联网上采集的判决文书，使用BIO的形式完成标记
    标记的实体包括: PER, LOC, ORG, DATE
    相比于数据源1，2，3多了 DATE 类型的实体
    :return:
    """
    print('Start Process data_4...')
    base_dir = '../data/data_4'
    # divide_train_dev_test(base_dir, base_dir, 'new_label_data.txt')
    divide_train_dev_test(base_dir, base_dir, 'all_label_data.txt')
    print('Done.')


def data_acm_clear():
    """
    数据来源：ACM敏捷协同管理
    将 BOSON命名实体识别数据 转换成 BIO 标签并保存到 all_data.txt 文件中
    并对 all_data.txt 文件里面的数据划分训练集、测试集和验证集
    原始数据标签如下：
        EPS: 项目群
        PROJECT: 项目
        PLANDEFINE: 计划定义
        WBSL: WBS
        TASK: 任务
        USER: 用户
        ORG: 组织机构
        ADDR: 地点
        DATE: 日期
    产生四个文件：all_data.txt  train.txt  test.txt  dev.txt
    :return:
    """
    base_dir = '../data/data_4'
    raw_path = os.path.join(base_dir, 'entity.txt')
    save_path = os.path.join(base_dir, 'all_data.txt')
    dict = {'EPS': 'EPS', 'PROJECT': 'PROJECT', 'PLANDEFINE': 'PLANDEFINE', 'WBSL': 'WBSL', 'TASK': 'TASK',
            'USER': 'PER', 'ORG': 'ORG', 'ADDR': 'LOC', 'DATE': 'DATE'}
    segment = ['。', '!', '!', '?', '？']
    print('Start process data_acm...')
    with open(save_path, 'w+', encoding='utf-8') as f_s:
        for line in open(raw_path, 'r', encoding='utf-8'):
            line = line.strip().replace(' ', '')
            line = line.replace('\\n\\n', '')
            line = line.replace('\t', '')
            line = line.replace('\xa0', '')
            index = 0
            length = len(line)
            sentences = []

            while True:
                if index >= length:
                    break

                if line[index] != "{":
                    sentences.append(line[index] + '\t' + 'O' + '\n')
                    index += 1
                elif line[index] == '{' and index + 2 < length and line[index + 1] == '{':
                    end_index = line[index:].find('}}')
                    if end_index == -1 or index + end_index >= length:
                        break

                    temp = line[index + 2: index + end_index]
                    if temp.find(":") == -1:
                        break

                    label_name = temp.split(":")
                    if len(label_name) != 2:
                        index = index + end_index + 2
                        continue
                    if label_name[0] in dict.keys():
                        result = get_data_label(dict[label_name[0]], label_name[1])
                        sentences.extend(result)
                    index = index + end_index + 2
                else:
                    break

            for word in sentences:
                f_s.write(word)
                if word[0] in segment:
                    f_s.write('\n')
            if len(sentences) != 0:
                last_word = sentences[-1]
                if last_word != "" and last_word[0] not in segment:
                    f_s.write('\n')
        divide_train_dev_test(base_dir, base_dir, 'all_data.txt')
        print('Done.')


def data_dict_acm_clear():
    """
    数据来源：ACM敏捷协同管理 新整理数据
    输入两个文件 dict_acm.txt, all_data.txt BOSON命名实体识别数据 转换成 BIO 标签
    产生三个个文件：  train.txt  test.txt  dev.txt
    :return:
    """
    base_dir = './data/data_acm'
    raw_path = os.path.join(base_dir, 'dict_acm.txt')
    save_path = os.path.join(base_dir, 'all_data.txt')

    with open(save_path, 'w+', encoding='utf-8') as f_s:
        for line in open(raw_path, 'r', encoding='utf-8'):
            line = line.strip().replace(' ', '')
            line = line.replace('\\n\\n', '')
            line = line.replace('\t', '')
            line = line.replace('\xa0', '')
            line = line.replace('{{', '')
            line = line.replace('}}', '')
            line = line.replace(']', '')
            list = line.split(',[')

            sentences = []

            if (len(list) == 2):
                label_name = list[0]
                label_list = list[1].split(',')
                for i in range(len(label_list)):
                    result = get_data_label(label_list[i], label_name)
                    sentences.extend(result)
                    sentences.extend('\n')

            elif (len(list) == 3):
                label_name = list[0]
                label_list = list[1].split(',')
                label_same = list[2].split(',')
                for i in range(len(label_list)):
                    # print(label_list[i])
                    # print(label_name)
                    result = get_data_label(label_list[i], label_name)
                    sentences.extend(result)
                    sentences.extend('\n')
                    for j in range(len(label_same)):
                        result = get_data_label(label_list[i], label_same[j])
                        sentences.extend(result)
                        sentences.extend('\n')

            # print(sentences)

            for word in sentences:
                f_s.write(word)
        divide_train_dev_test(base_dir, base_dir, 'all_data.txt')
        print("done...")


def data_new_acm_clear():
    """
    数据来源：ACM敏捷协同管理 新整理数据
    输入两个文件 new_acm_data.txt, all_data.txt BOSON命名实体识别数据 转换成 BIO 标签
    产生三个个文件：  train.txt  test.txt  dev.txt
    :return:
    """
    base_dir = './data/data_acm_new'
    raw_path = os.path.join(base_dir, 'new_acm_data.txt')
    save_path = os.path.join(base_dir, 'all_data.txt')

    with open(save_path, 'w+', encoding='utf-8') as f_s:
        for line in open(raw_path, 'r', encoding='utf-8'):
            # print("-----readline-----")
            line = line.strip().replace(' ', '')
            line = line.replace('\\n\\n', '')
            line = line.replace('\n', '')
            line = line.replace('\t', '')
            line = line.replace('\xa0', '')
            # print(line)
            index = 0
            sentences = []
            while index < len(line):
                if line[index] != "{":
                    if line[index] != "":
                        sentences.append(line[index] + '\t' + 'O' + '\n')
                    index += 1
                elif line[index] == '{' and index + 2 < len(line) and line[index + 1] == '{':
                    end_index = line[index:].find('}}')
                    if end_index == -1 or index + end_index >= len(line):
                        break

                    temp = line[index + 2: index + end_index]
                    # print("temp:",temp)

                    if temp.find(",[") == -1:
                        break

                    label_name = temp.split(",[")
                    # print("label_name:",label_name)
                    labels =[]
                    names=[]

                    if len(label_name) < 2:
                        index = index + end_index + 2
                        continue
                    if len(label_name) == 3:
                        names.append(label_name[0])

                        if label_name[2].find(",") == -1:
                            name = label_name[2].replace("]","")
                            names.append(name)
                        else:
                            name = label_name[2].replace("]","").split(",")
                            names.extend(name)
                        if label_name[1].find(",") == -1:
                            label = label_name[1].replace("]","")
                            labels.append(label)
                        else:
                            label = label_name[1].replace("]","").split(",")
                            labels.extend(label)

                    if len(label_name) == 2:
                        names.append(label_name[0])
                        if label_name[1].find(",") == -1:
                            label = label_name[1].replace("]", "")
                            labels.append(label)
                        else:
                            label = label_name[1].replace("]","").split(",")
                            labels.extend(label)


                    # print("labels",labels)
                    # print("names",names)

                    for i in range (len(labels)):
                        for j in range (len(names)):
                            result = get_data_label(str(labels[i]), str(names[j]))
                            sentences.extend(result)
                            sentences.extend('\n')
                    index = index + end_index + 2
                else:
                    break

            # print(sentences)

            for word in sentences:
                f_s.write(word)
        divide_train_dev_test(base_dir, base_dir, 'all_data.txt')
        print("done...")


def join_all_data(mode):
    """
    合并三个数据源下面的 mode 数据
    :param mode: 分为三个模式：train dev test
    :return:
    """
    print('Begin combine {0} data...'.format(mode))
    sen1 = load_sentences(os.path.join('./data/data_ori/', mode + '.txt'))
    sen5 = load_sentences(os.path.join('./data/data_acm_new/', mode + '.txt'))

    # f_w = open(os.path.join('../data/train_data', mode + '.txt'), 'w+', encoding='utf-8')
    f_w = open(os.path.join('./data/data_train', 'example.' + mode), 'w+', encoding='utf-8')

    mode_sen = []
    mode_sen.extend(sen1)
    mode_sen.extend(sen5)
    print("{0} sentences num is {1}".format(mode, len(mode_sen)))

    for sen in mode_sen:
        for word in sen:
            f_w.write(word[0] + ' ' + word[1] + '\n')
        f_w.write('\n')
    f_w.close()
    print('Done.')


if __name__ == '__main__':
    data_ori_clear()
    # data_dict_acm_clear()
    data_new_acm_clear()
    join_all_data(mode='train')
    join_all_data(mode='test')
    join_all_data(mode='dev')

    # data_new_acm_clear()
