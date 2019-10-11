import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


class data_adress():
    """数据处理"""
    def __init__(self):
        """初始化需要的参数"""
        self.x_train = tf.placeholder(tf.float32, shape=[None, 5000*12])  # 训练样本的特征值
        self.y_train = tf.placeholder(tf.float32, shape=[None, 55])  # 训练样本的目标值
        self.label_path = "./test_data/hefei_data/hf_round1_label.txt"  # label的路径
        self.label_column_names = ["id", "age", "gender", "symptom1", "symptom2", "symptom3", "symptom4", "symptom5",
                                      "symptom6", "symptom7", "symptom8"]  # label的列标签名字
        self.hf_terecords_data = "./hf_terecords"  # 存储训练数据样本和标签的tfrecords文件夹
    def label(self):
        # 标签数据处理
        hf_label = pd.read_csv(self.label_path, sep="\t", names=self.label_column_names)  # 标签数据
        gender = {"FEMALE": 0, "MALE": 1}  # 定义map，女性为0，男性为1
        hf_label["gender"] = hf_label["gender"].map(gender)
        hf_label[["age", "gender"]] = hf_label[["age", "gender"]].fillna(-1)  # 处理nan值，缺失值填充-1
        age = hf_label.groupby(["age"]).count()["symptom2"]
        symptom1 = hf_label.groupby(["symptom1"]).count()
        arrythmia = pd.read_table("./test_data/hefei_data/hf_round1_arrythmia.txt", names=["arrythmia"])  # 处理异常数据
        arrythmia_num_list = arrythmia["arrythmia"].unique()  # 异常数据的类别总共有多少
        arrythmia_label = pd.DataFrame(np.zeros([hf_label.shape[0], len(arrythmia_num_list)]),
                                       columns=arrythmia_num_list)  # 将异常特征组成dataframe
        # print(arrythmia_label)
        for j in range(8):  # 将每个样本中含有的异常特征
            for i in range(hf_label.shape[0]):
                arrythmia_label.loc[i, hf_label["symptom{}".format(j + 1)][i]] = 1
        label_num = pd.concat([hf_label, arrythmia_label], axis=1)
        label = label_num[arrythmia_num_list.tolist()]  # 选择需要的目标列
        # print(label)
        # label = label.as_matrix()
        label = label.values
        return label, arrythmia_num_list

    def data_visual(self):
        # 数据可视化,由于缺失值太多，本次训练中只考虑
        label, arrythmia_num_list = self.label()  # 特征dataframe
        # x = [-1,0,10,30,50,70,90,114]
        x = label["age"]
        age = label.groupby(["age"]).sum()

        for i in range(len(arrythmia_num_list)):
            y = label[arrythmia_num_list[i]]
            title = arrythmia_num_list[i]
            plt.bar(x=x, height=y)
            plt.show()

    def address_train_data(self):
        # 训练数据
        # label, arrythmia_num_list = self.label()  # 特征dataframe
        train_file_names = os.listdir("./test_data/hefei_data/train")  # 训练数据的文件名
        train_file_names.sort(key=lambda x: int(x[:-4]))  # 对训练文件进行排排序
        train_data = []  # 每个样本的训练数据
        for filename in train_file_names:
            # 读取每个样本的训练数据，并计算出其他几个数据
            id_name = "./test_data/hefei_data/train/" + filename
            y_train = pd.read_csv(id_name, sep=" ")
            # III = II - I
            # aVR = -(I + II) / 2
            # aVL = I - II / 2
            # aVF = II - I / 2
            y_train["III"] = y_train["II"] - y_train["I"]
            y_train["aVR"] = -(y_train["I"] + y_train["II"]) / 2
            y_train["aVL"] = (y_train["I"] - y_train["II"]) / 2
            y_train["aVF"] = (y_train["II"] - y_train["I"]) / 2
            names = y_train.columns.values
            # print(names)
            # for name in names:
            #     print(name)
            #     print(y_train[name].max())
            #     y_train.apply(lambda x: (x - y_train[name].min(x)) / (y_train[name].max(x) - y_train[name].min(x)))
            train_data.append(y_train[names])
        # train_data = train_data.values
        tt = train_data[1]
        return train_data

    def save_tfrecords(self):
        """将样本和样本对应的特征保存在一起,保存为tfrecords类型文件，方便读取"""
        label, arrythmia_num_list = self.label()  # 标签数据
        train_data = self.address_train_data()  # 训练数据
        with tf.python_io.TFRecordWriter(self.hf_terecords_data) as tfrecords_writer:
            # tfrecords文件写入器,使用上下文环境，方便关闭writer、
            for i in range(len(train_data)):  # 总共的样本数，每个train_data[i]的数据格式为5000*12
                x_train = train_data[i].astype(np.float32).values.tostring()  # 取出训练数据的每个id数据，去掉行列的标签
                y_train = label[i].astype(np.float32).tostring()  # 样本对应的标签数据
                print(y_train)
                example = tf.train.Example(features=tf.train.Features(feature={
                    "x_train":tf.train.Feature(bytes_list=tf.train.BytesList(value=[x_train])),
                    "y_train":tf.train.Feature(int64_list=tf.train.Int64List(value=[y_train]))
                }))  # 构造每个样本的example协议
                tfrecords_writer.write(example.SerializeToString())  # 写入tfrecords

    def read_hf_tfcords(self):
        """读取保存的tfcords文件"""
        # 文件队列
        hf_tfrecords_queue = tf.train.string_input_producer(["./hf_train_data.tfrecords"])
        # tfcords读写器
        tfrecords_reader = tf.TFRecordReader()
        # 读取内容
        key, value = tfrecords_reader.read(hf_tfrecords_queue)
        # 解析example
        features = tf.parse_single_example(value,features={
            "x_train": tf.FixedLenFeature([], tf.string),
            "y_train": tf.FixedLenFeature([], tf.string)
        })
        # 解码string类型
        x_train = tf.reshape(tf.decode_raw(features["x_train"], tf.float32), [5000*12])
        y_train = tf.reshape(tf.decode_raw(features["y_train"], tf.float32),[55])
        # 批量处理样本数据
        x_batch_train, y_batch_train = tf.train.shuffle_batch([x_train, y_train], batch_size=50, capacity=2000, min_after_dequeue=1000,num_threads=4)

        return x_batch_train, y_batch_train

    def weight_bias(self,weight_shape, bias_shape):
        """
           卷积层的weight和bias
           :param weight_shape:weight形状
           :param bias_shape: bias形状
           :return: weight bias
        """
        weight = tf.random_normal(weight_shape)  # 权重
        bias = tf.random_normal(bias_shape)  # 偏置
        return weight,bias

    def conv(self):
        # 进行卷积层优化
        weight1, bias1 = self.weight_bias([5,5,1,32],[32])  # 第一次卷积的weight and bias
        x1 = tf.reshape(self.x_train, [-1, 5000,12,1])  # 修改特征值的形状和weight对应
        conv1 = tf.nn.conv2d(x1,filter=weight1,strides=[1,1,1,1],padding="SAME",name="conv1")  # 卷积
        relu1 = tf.nn.relu(features=conv1)  # relu激活函数
        pool1 = tf.nn.max_pool(relu1,ksize=[1,20,1,1], strides=[1,1,1,1],padding="SAME")  # 池化

        weight2, bias2 = self.weight_bias([5, 5, 32, 64], [64])  # 第一次卷积的weight and bias
        conv2 = tf.nn.conv2d(pool1, filter=weight1, strides=[1, 1, 1, 1], padding="SAME", name="conv1")  # 卷积
        relu2 = tf.nn.relu(features=conv2)  # relu激活函数
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 20, 1, 1], strides=[1, 1, 1, 1], padding="SAME")  # 池化
        return pool2

    def tf_data(self):
        # 准备tensorflow数据占位符,每个样本5000rows*12columns  55个目标值
        x_train = self.conv()
        x_train = tf.reshape(x_train,[-1,x_train.shape[1]*x_train.shape[2]*x_train.shape[3]])
        y_train = self.y_train
        weights = tf.Variable(tf.random_normal([5000* 12, 55]))  # 随机权重值
        bias = tf.Variable(tf.random_normal([55]))  # 随机偏重值
        y_predict = tf.matmul(x_train, weights) + bias  # 进行全链接计算
        loss = tf.nn.l2_loss(tf.subtract(y_predict, y_train))  # 计算误差
        train_loss = tf.train.GradientDescentOptimizer(0.0000000001).minimize(loss)  # 进行梯度下降，降低误差
        init_v = tf.global_variables_initializer()  # 初始化变量
        print("*"*10)
        x_batch_train, y_batch_train = self.read_hf_tfcords()
        # print(x_batch_train)
        with tf.Session() as sess:
            sess.run(init_v)  # 初始化变量
            coord = tf.train.Coordinator()  # 线程协调器
            thread = tf.train.start_queue_runners(coord=coord, sess=sess)
            for i in range(40000):
                print("-"*10)
                # sess.run(train_loss)
                # print(x_batch_train, y_batch_train)
                x_batch, y_batch = sess.run([x_batch_train, y_batch_train])
                # print(x_batch)
                feed_dict = {
                    x_train: x_batch,
                    y_train: y_batch}
                sess.run(train_loss, feed_dict=feed_dict)
                print(sess.run(loss,feed_dict=feed_dict))
            coord.request_stop()
            coord.join(thread)

if __name__ == '__main__':
    data_adress = data_adress()
    data_adress.tf_data()