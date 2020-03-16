# _*_coding:utf-8_*_
import tensorflow as tf
from tensorflow.python.framework import graph_util
# from create_tf_record import *
import os
import json
import pickle
from ChineseNER.utils import get_logger, make_path, clean, create_model, save_model
from ChineseNER.utils import print_config, save_config, load_config, test_ner
from ChineseNER.model import Model
from ChineseNER.data_utils import load_word2vec, create_input, input_from_line, BatchManager

# resize_height = 224  # 指定图片高度
# resize_width = 224  # 指定图片宽度

CONFIG_DIR = './output/config_file'  # 模型参数配置文件
MODEL_PKL_PATH = './output/maps.pkl'  # .pkl文件
MODEL_CKPT_PATH = './output/ckpt/'  # ckpt模型文件路径
LOG_FILE = './output/log/train.log'

MODEL_SAVE_PATH_NER = './output/model_server/ner_model/'
MODEL_VERSION_NER = '1'
MODEL_SIGNATURE_NER = 'prediction_labels_longtext'


def freeze_graph(input_checkpoint, output_graph):
    '''

    :param input_checkpoint:
    :param output_graph:  PB 模型保存路径
    :return:
    '''
    # 检查目录下ckpt文件状态是否可用
    # checkpoint = tf.train.get_checkpoint_state(model_folder)
    # 得ckpt文件路径
    # input_checkpoint = checkpoint.model_checkpoint_path

    # 指定输出的节点名称，该节点名称必须是元模型中存在的节点
    # output_node_names = "InceptionV3/Logits/SpatialSqueeze"
    output_node_names = "optimizer/char_BiLSTM/bidirectional_rnn/bw/coupled_input_forget_gate_lstm_cell/_w_hi/Adam_1"
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()  # 获得默认的图
    input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
        # 模型持久化，将变量值固定
        output_graph_def = graph_util.convert_variables_to_constants(
            sess=sess,
            # 等于:sess.graph_def
            input_graph_def=input_graph_def,
            # 如果有多个输出节点，以逗号隔开
            output_node_names=output_node_names.split(","))

        # 保存模型
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())  # 序列化输出
        # 得到当前图有几个操作节点
        print("%d ops in the final graph." % len(output_graph_def.node))

        # for op in graph.get_operations():
        #     print(op.name, op.values())



def load_and_save_model():
    """
    将模型保存成pb格式
    :return:
    """
    config = load_config(CONFIG_DIR)

    batch_size = 1

    with open(os.path.join(MODEL_PKL_PATH), 'rb') as rf:
        # label_list = pickle.load(rf)
        char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(rf)
        # print("char_to_id",char_to_id)
        # print("id_to_char",id_to_char)
        # print("tag_to_id",tag_to_id)
        # print("id_to_tag",id_to_tag)

    # num_labels = len(label_list) + 1

    logger = get_logger(LOG_FILE)

    tf.reset_default_graph()

    # input_ids_p = tf.placeholder(tf.int32, [batch_size, config["max_seq_length"]],name="input_ids")
    # input_mask_p = tf.placeholder(tf.int32, [batch_size, config["max_seq_length"]],name="input_mask")
    # label_ids_p = tf.placeholder(tf.int32, [batch_size, config["max_seq_length"]],name="label_ids")
    # segment_ids_p = tf.placeholder(tf.int32, [batch_size, config["max_seq_length"]],name="segment_ids")

    # # add placeholders for the model
    # input_ids_p = tf.placeholder(dtype=tf.int32, shape=[None, None], name="input_ids")
    # input_mask_p = tf.placeholder(dtype=tf.int32,shape=[None, None],name="input_mask")
    # label_ids_p = tf.placeholder(dtype=tf.int32, shape=[None, None], name="label_ids")
    # segment_ids_p = tf.placeholder(dtype=tf.int32, shape=[None, None], name="segment_ids")

    # bert_config = modeling.BertConfig.from_json_file('../chinese_L-12_H-768_A-12/bert_config.json')
    # (_, _, _, pred_ids) = create_model(bert_config, config["is_training"], input_ids_p,
    #                                    input_mask_p, segment_ids_p, label_ids_p,
    #                                    num_labels, config["use_one_hot_embeddings"])

    ckpt_file = tf.train.latest_checkpoint(MODEL_CKPT_PATH)
    print('Load model from {}. \n'.format(ckpt_file))
    # saver = tf.train.Saver()

    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth = True
    sess = tf.Session(config=gpu_config)

    model = create_model(sess, Model, MODEL_CKPT_PATH, load_word2vec, config, id_to_char, logger)
    print(model)

    # input_line =
    # pred_ids= model.evaluate_line(sess, input_from_line(input_line, char_to_id), id_to_tag)

    input_ids_p = model.char_inputs
    label_ids_p = model.targets
    segment_ids_p = model.seg_inputs


    # result = pred_ids.evaluate_line(sess, input_from_line(line, char_to_id), id_to_tag)


    saver = model.saver


    saver.restore(sess, ckpt_file)
    print("Load model successful!")

    export_path = os.path.join(tf.compat.as_bytes(MODEL_SAVE_PATH_NER), tf.compat.as_bytes(MODEL_VERSION_NER))
    print("model export path: {0}".format(export_path))

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    input_tensor_ids_info = tf.saved_model.utils.build_tensor_info(input_ids_p)
    # input_tensor_mask_info = tf.saved_model.utils.build_tensor_info(input_mask_p)
    input_tensor_segment_info = tf.saved_model.utils.build_tensor_info(segment_ids_p)
    input_tensor_labels_info = tf.saved_model.utils.build_tensor_info(label_ids_p)

    output_tensor_info = tf.saved_model.utils.build_tensor_info(saver)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                'input_ids': input_tensor_ids_info,
                # 'input_mask': input_tensor_mask_info,
                'segment_ids': input_tensor_segment_info,
                'label_ids': input_tensor_labels_info
            },
            outputs={
                'pred_ids': output_tensor_info
            },
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
        )
    )

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            MODEL_SIGNATURE_NER: prediction_signature,
        }
    )

    builder.save()
    print('Save model to be Serving success.')


def restore_and_save(input_checkpoint, export_path_base):
    checkpoint_file = tf.train.latest_checkpoint(input_checkpoint)
    graph = tf.Graph()

    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)

        with sess.as_default():
            # 载入保存好的meta graph，恢复图中变量，通过SavedModelBuilder保存可部署的模型
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)
            print("***",graph.get_name_scope())

            export_path_base = export_path_base
            export_path = os.path.join(
                tf.compat.as_bytes(export_path_base),
                tf.compat.as_bytes(str(count)))
            print('Exporting trained model to', export_path)
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            # 建立签名映射，需要包括计算图中的placeholder（ChatInputs, SegInputs, Dropout）和我们需要的结果（project/logits,crf_loss/transitions）
            """
            build_tensor_info：建立一个基于提供的参数构造的TensorInfo protocol buffer，
            输入：tensorflow graph中的tensor；
            输出：基于提供的参数（tensor）构建的包含TensorInfo的protocol buffer
                        get_operation_by_name：通过name获取checkpoint中保存的变量，能够进行这一步的前提是在模型保存的时候给对应的变量赋予name
            """

            char_inputs = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("ChatInputs").outputs[0])
            seg_inputs = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("SegInputs").outputs[0])
            dropout = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("Dropout").outputs[0])
            logits = tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name("project/Reshape").outputs[0])

            transition_params = tf.saved_model.utils.build_tensor_info(
                graph.get_operation_by_name("crf_loss/Mean").outputs[0])
                # graph.get_operation_by_name("crf_loss/transitions").outputs[0])  # 原代码,修改无变化

            """
            signature_constants：SavedModel保存和恢复操作的签名常量。
            在序列标注的任务中，这里的method_name是"tensorflow/serving/predict"
            """
            # 定义模型的输入输出，建立调用接口与tensor签名之间的映射
            labeling_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs={
                        "charinputs":
                            char_inputs,
                        "dropout":
                            dropout,
                        "seginputs":
                            seg_inputs,
                    },
                    outputs={
                        "logits":
                            logits,
                        "transitions":
                            transition_params
                    },
                    method_name="tensorflow/serving/predict"))

            """
            tf.group : 创建一个将多个操作分组的操作，返回一个可以执行所有输入的操作
            """
            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

            """
            add_meta_graph_and_variables：建立一个Saver来保存session中的变量，
                                          输出对应的原图的定义，这个函数假设保存的变量已经被初始化；
                                          对于一个SavedModelBuilder，这个API必须被调用一次来保存meta graph；
                                          对于后面添加的图结构，可以使用函数 add_meta_graph()来进行添加
            """
            # 建立模型名称与模型签名之间的映射
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                # 保存模型的方法名，与客户端的request.model_spec.signature_name对应
                signature_def_map={
                    tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        labeling_signature},
                legacy_init_op=legacy_init_op)

            builder.save()
            print("Build Done")

if __name__ == '__main__':

    # # 输入ckpt模型路径
    # input_checkpoint = './output/ckpt/ner.ckpt'
    #
    # # 输出pb模型的路径
    # out_pb_path = "./output/frozen_model.pb"
    #
    # # 调用freeze_graph将ckpt转为pb
    # freeze_graph(input_checkpoint, out_pb_path)

    # # TensorFlow Serving Model
    # if os.path.exists(MODEL_SAVE_PATH_NER):
    #     ner_versions = []
    #     for version_name in os.listdir(MODEL_SAVE_PATH_NER):
    #         ner_versions.append(version_name)
    #     if MODEL_VERSION_NER not in ner_versions:
    #         load_and_save_model()
    #     del ner_versions
    # else:
    #     load_and_save_model()

    ### 测试模型转换
    tf.flags.DEFINE_string("ckpt_path", "./output/ckpt", "path of source checkpoints")
    tf.flags.DEFINE_string("pb_path", "./output/model_server", "path of servable models")
    tf.flags.DEFINE_integer("version", 1, "the number of model version")
    tf.flags.DEFINE_string("classes", 'ner_model', "multi-models to be converted")
    FLAGS = tf.flags.FLAGS

    classes = FLAGS.classes
    input_checkpoint = FLAGS.ckpt_path + "/"
    model_path = FLAGS.pb_path + '/' + classes
    print("model_path:", model_path)

    # 版本号控制
    count = FLAGS.version
    modify = False
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    else:
        for v in os.listdir(model_path):
            print(type(v), v)
            if int(v) >= count:
                count = int(v)
                modify = True
        if modify:
            count += 1

    # 模型格式转换
    restore_and_save(input_checkpoint, model_path)
