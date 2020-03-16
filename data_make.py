## 随机插入样例数据到原有训练数据中测试训练结果,
import random
import re


class DataMake:
    def __init__(self):
        self.RANDOM_NUM = 3

    def rand_datamake(self, acmdata_path, textdata_path):
        with open(textdata_path, 'r', encoding='UTF-8') as f:
            str_lines = f.read()
            sen_list = re.split('(。|！|\!|？|\?|\（|\）|\\n|\\t)', str_lines)  # 保留分割符
            # print("***sen_list_old***:", sen_list)
            with open(acmdata_path, 'r', encoding='UTF-8') as f:
                acm_line = f.read()
                line_list = acm_line.split("\n")

            for i in range(len(line_list)):
                str_num = random.randint(0, len(sen_list) - 1)
                # print("str_num:", str_num)
                print(sen_list[str_num])
                # acm_num = random.randint(0, len(line_list) - 1)
                # print("acm_num:", acm_num)
                print(line_list[i])
                sen_list[str_num] = sen_list[str_num] + line_list[i]
            # print("***sen_list***:", sen_list)
            new_data = "".join(sen_list)

            print(new_data)
        return new_data



if __name__ == '__main__':
    acmdata_path = "./data/data_acm/dict_acm.txt"   # 标签数据
    textdata_path = "./data/data_test/test_data.txt"  # 干扰数据
    test = DataMake()
    new_text = test.rand_datamake(acmdata_path, textdata_path)
    with open("./data/data_acm_new/new_acm_data.txt", "w", encoding='utf-8') as f:
        f.write(new_text)

