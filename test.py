# with open("data/sentence.txt" ,'r' ,encoding='UTF-8') as f:
#     line = f.read()
#     print(line)
#     print(type(line))

# s = [['安', 'B-WBSL'], ['全', 'I-WBSL'], ['帽', 'I-WBSL'], ['、', 'I-WBSL'], ['安', 'I-WBSL'], ['全', 'I-WBSL'], ['带', 'I-WBSL'], ['等', 'I-WBSL'], ['防', 'I-WBSL'], ['护', 'I-WBSL'], ['用', 'I-WBSL'], ['品', 'I-WBSL'], ['的', 'I-WBSL'], ['发', 'I-WBSL'], ['放', 'I-WBSL'], ['及', 'I-WBSL'], ['使', 'I-WBSL'], ['情', 'I-WBSL'], ['况', 'E-WBSL'], ['属', 'O'], ['于', 'O'], ['桃', 'B-PROJECT'], ['李', 'I-PROJECT'], ['春', 'I-PROJECT'], ['风', 'I-PROJECT'], ['苑', 'I-PROJECT'], ['（', 'I-PROJECT'], ['黄', 'I-PROJECT'], ['鹏', 'I-PROJECT'], ['项', 'I-PROJECT'], ['目', 'I-PROJECT'], ['组', 'I-PROJECT'], ['）', 'I-PROJECT'], [',', 'O'], ['也', 'O'], ['属', 'O'], ['于', 'O'], ['安', 'B-PLANDEFINE'], ['全', 'I-PLANDEFINE'], ['文', 'I-PLANDEFINE'], ['明', 'I-PLANDEFINE'], ['施', 'I-PLANDEFINE'], ['工', 'I-PLANDEFINE'], ['检', 'I-PLANDEFINE'], ['查', 'I-PLANDEFINE'], ['计', 'I-PLANDEFINE'], ['划', 'E-PLANDEFINE'], [',', 'O'], ['也', 'O'], ['属', 'O'], ['于', 'O']]
# # s= [['房', 'B-TASK'], ['子', 'I-TASK'], ['（', 'I-TASK'], ['±', 'I-TASK'], ['0', 'I-TASK'], ['）', 'E-TASK']]
# tag_to_id = {'B-EPS': 23, 'B-TASK': 18, 'B-PER': 14, 'E-ORG': 17, 'E-LOC': 28, 'E-WBSL': 7, 'B-ORG': 16, 'B-PROJECT': 10, 'S-WBSL': 29, 'S-TASK': 26, 'E-PER': 15, 'B-LOC': 27, 'I-PLANDEFINE': 3, 'E-EPS': 24, 'E-DATE': 21, 'I-PROJECT': 2, 'I-EPS': 22, 'I-LOC': 25, 'I-ORG': 5, 'I-TASK': 9, 'B-WBSL': 6, 'I-WBSL': 0, 'I-PER': 4, 'B-DATE': 20, 'E-TASK': 19, 'E-PROJECT': 11, 'B-PLANDEFINE': 12, 'I-DATE': 8, 'O': 1, 'E-PLANDEFINE': 13}
#
# tags = [tag_to_id[w[-1]] for w in s]
#
# print(tags)

# # 获取当前路径
# import os
# base_path = os.path.dirname(os.path.abspath(__file__))
# print(base_path)


# 测试替换特殊字符

# s="}}  起重吊装登高作业人员安全防护情况}}  施工机具防护、加工区防护、安全通道设置}}"
# print(len(s))
# s=s.replace("}"," ")
# print(len(s))
# print(s)

# sentences=[[['海', 'O'], ['钓', 'O'], ['比', 'O'], ['赛', 'O'], ['地', 'O'], ['点', 'O'], ['在', 'O'], ['厦', 'B-LOC'], ['门', 'I-LOC'], ['与', 'O'], ['金', 'B-LOC'], ['门', 'I-LOC'], ['之', 'O'], ['间', 'O'], ['的', 'O'], ['海', 'O'], ['域', 'O'], ['。', 'O']]]
# print(type(sentences))
# # print(sentences)
# tags = [[char[-1] for char in s] for s in sentences]
# print(tags)
#
# def create_dico(item_list):
#     """
#     Create a dictionary of items from a list of list of items.
#     """
#     assert type(item_list) is list
#     dico = {}
#     for items in item_list:
#         for item in items:
#             if item not in dico:
#                 dico[item] = 1
#             else:
#                 dico[item] += 1
#     return dico
#
# dico = create_dico(tags)
# print(dico)
#
# # tensorboard查找输出的节点名称
# from tensorflow.python import pywrap_tensorflow
#
# checkpoint_path = './output/ckpt/ner.ckpt'
# reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
# var_to_shape_map = reader.get_variable_to_shape_map()
# for key in var_to_shape_map:
#     print('tensor_name: ', key)


# # coding:utf-8
# import sys, os, io
# import tensorflow as tf
#
#
# def restore_and_save(input_checkpoint, export_path_base):
#     checkpoint_file = tf.train.latest_checkpoint(input_checkpoint)
#     graph = tf.Graph()
#
#     with graph.as_default():
#         session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
#         sess = tf.Session(config=session_conf)
#
#         with sess.as_default():
#             # 载入保存好的meta graph，恢复图中变量，通过SavedModelBuilder保存可部署的模型
#             saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
#             saver.restore(sess, checkpoint_file)
#             print("***",graph.get_name_scope())
#
#             export_path_base = export_path_base
#             export_path = os.path.join(
#                 tf.compat.as_bytes(export_path_base),
#                 tf.compat.as_bytes(str(count)))
#             print('Exporting trained model to', export_path)
#             builder = tf.saved_model.builder.SavedModelBuilder(export_path)
#
#             # 建立签名映射，需要包括计算图中的placeholder（ChatInputs, SegInputs, Dropout）和我们需要的结果（project/logits,crf_loss/transitions）
#             """
#             build_tensor_info：建立一个基于提供的参数构造的TensorInfo protocol buffer，
#             输入：tensorflow graph中的tensor；
#             输出：基于提供的参数（tensor）构建的包含TensorInfo的protocol buffer
#                         get_operation_by_name：通过name获取checkpoint中保存的变量，能够进行这一步的前提是在模型保存的时候给对应的变量赋予name
#             """
#
#             char_inputs = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("ChatInputs").outputs[0])
#             seg_inputs = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("SegInputs").outputs[0])
#             dropout = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("Dropout").outputs[0])
#             logits = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("project/Reshape").outputs[0])
#
#             transition_params = tf.saved_model.utils.build_tensor_info(
#                 graph.get_operation_by_name("crf_loss/Mean").outputs[0])
#
#             """
#             signature_constants：SavedModel保存和恢复操作的签名常量。
#             在序列标注的任务中，这里的method_name是"tensorflow/serving/predict"
#             """
#             # 定义模型的输入输出，建立调用接口与tensor签名之间的映射
#             labeling_signature = (
#                 tf.saved_model.signature_def_utils.build_signature_def(
#                     inputs={
#                         "charinputs":
#                             char_inputs,
#                         "dropout":
#                             dropout,
#                         "seginputs":
#                             seg_inputs,
#                     },
#                     outputs={
#                         "logits":
#                             logits,
#                         "transitions":
#                             transition_params
#                     },
#                     method_name="tensorflow/serving/predict"))
#
#             """
#             tf.group : 创建一个将多个操作分组的操作，返回一个可以执行所有输入的操作
#             """
#             legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
#
#             """
#             add_meta_graph_and_variables：建立一个Saver来保存session中的变量，
#                                           输出对应的原图的定义，这个函数假设保存的变量已经被初始化；
#                                           对于一个SavedModelBuilder，这个API必须被调用一次来保存meta graph；
#                                           对于后面添加的图结构，可以使用函数 add_meta_graph()来进行添加
#             """
#             # 建立模型名称与模型签名之间的映射
#             builder.add_meta_graph_and_variables(
#                 sess, [tf.saved_model.tag_constants.SERVING],
#                 # 保存模型的方法名，与客户端的request.model_spec.signature_name对应
#                 signature_def_map={
#                     tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
#                         labeling_signature},
#                 legacy_init_op=legacy_init_op)
#
#             builder.save()
#             print("Build Done")
#
#
# ### 测试模型转换
# tf.flags.DEFINE_string("ckpt_path", "./output/ckpt", "path of source checkpoints")
# tf.flags.DEFINE_string("pb_path", "./output/model_server", "path of servable models")
# tf.flags.DEFINE_integer("version", 1, "the number of model version")
# tf.flags.DEFINE_string("classes", 'ner_model', "multi-models to be converted")
# FLAGS = tf.flags.FLAGS
#
# classes = FLAGS.classes
# input_checkpoint = FLAGS.ckpt_path + "/"
# model_path = FLAGS.pb_path + '/' + classes
# print("model_path:",model_path)
#
# # 版本号控制
# count = FLAGS.version
# modify = False
# if not os.path.exists(model_path):
#     os.mkdir(model_path)
# else:
#     for v in os.listdir(model_path):
#         print(type(v), v)
#         if int(v) >= count:
#             count = int(v)
#             modify = True
#     if modify:
#         count += 1
#
# # 模型格式转换
# restore_and_save(input_checkpoint, model_path)


####grpc测试代码s
# import grpc
# from tensorflow_serving.apis import predict_pb2
# from tensorflow_serving.apis import prediction_service_pb2_grpc
# import tensorflow as tf
# import numpy as np
#
# # 将文本进行预处理，得到list格式的输入数据，然后将其转为tensor后序列化
# def get_input(sentence):
#
#     charinputs = np.reshape().tolist()
#     seginputs = np.reshape().tolist()
#
#     return charinputs, seginputs
# # # 后处理
# # def get_output(logits,transitions):
# #
# #     return tags
#
# def test_grpc(sentence):
#     ### 建立连接
#     channel = grpc.insecure_channel("localhost:8500")
#     stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
#     request = predict_pb2.PredictRequest()
#
#     # 指定启动tensorflow serving时配置的model_name和是保存模型时的方法名
#     request.model_spec.name = "model1"
#     request.model_spec.signature_name = "serving_default"
#
#     ### 构造输入tensor
#     # 将文本进行预处理，得到list格式的输入数据，然后将其转为tensor后序列化
#     # charinputs, seginputs = get_input(sentence)
#     charinputs, seginputs = get_input(sentence)
#
#     request.inputs["charinputs"].ParseFromString(tf.contrib.util.make_tensor_proto(charinputs, dtype=tf.int32).SerializeToString())
#     request.inputs["seginputs"].ParseFromString(tf.contrib.util.make_tensor_proto(seginputs, dtype=tf.int32).SerializeToString())
#     request.inputs["dropout"].ParseFromString(tf.contrib.util.make_tensor_proto(0.5, dtype=tf.float32).SerializeToString())
#
#     ### 获取模型输入结果
#     # response = stub.Predict(request, timeout)
#     response = stub.Predict.future(request, 2.0)
#
#     results = {}
#     for key in response.outputs:
#         tensor_proto = response.outputs[key]
#         results[key] =  tf.contrib.util.make_ndarray(tensor_proto)
#
#     # 从results中取所需要的结果，不一定是这两个变量哦
#     logits = results["logits"]
#     transitions = results["transitions"]
#
#     # 后处理
#     # tags = get_output(logits, transitions)
#     print("logits:",logits)
#     print("transitions:",transitions)
#
#
# if __name__ == '__main__':
#     with open("data/sentence.txt", 'r', encoding='UTF-8') as f:
#         line = f.read()
#         test_grpc(line)


# ## 计算标签个数
# import re
#
# with open("./data/data_acm/dict_acm.txt", 'r', encoding='UTF-8') as f:
#     line = f.read()
#     line_list = line.split("\n")
#     label_list = []
#     for i in range(len(line_list)):
#         line_list_split = line_list[i].split(",[")
#         # print(line_list_split)
#         label = line_list_split[1]
#         label = label.replace('}}', '')
#         label = label.replace(']', '')
#         print(label)
#         labels = label.split(",")
#         for j in range(len(labels)):
#             label_list.append(labels[j])
#
# label_list = list(set(label_list))
# # print(line_list)
# print(label_list)
# print(len(label_list))
#
# ##['DEPT', 'MEETING', 'WBSL', 'TASK', 'RSRCEQUIP', 'PROJECT', 'PROJDOC', 'PLANDEFINE', 'PREPA', 'MEETINGACTION', 'USER', 'RSRCMATERIAL', 'PROJORG', 'RSRCUSER', 'PROJDELV', 'BASELINE', 'PROBLEM', 'EPS']
# ##18




