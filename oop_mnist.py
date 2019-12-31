'''
手写数字识别 MNIST数据集 60000个手写字母的训练样本 10000个测试样本
超参数的确认  对于全连接网络隐藏层最好不要超过3层
mnist数据集的每个训练数据是28*28像素的图片 输入层节点应该为784(每一个作为一个输入特征)对应一个神经元
0-9个数字 10个分类 输出层为10个节点 每个神经元的输出对应一个类别的预测值 最大的值就是模型的预测结果
隐藏层不好确定 有几个经验公式  m = sqrt(n+l) + a | m = log2n | m = sqrt(n*l)
m是隐藏层节点数  n是输入层节点 l是输出层节点数  a是1到10之间的常数
3层 784 * 300 * 10的全连接神经网络 总共有300 * (784 + 1) + 10 * (300 + 1) = 238510个参数(w)
计算识别错误率 = 错误预测样本数 / 总样本数
'''

import struct
from full_connected_neural_network import NeuralNetwork
from datetime import datetime

# 全局变量
input_layer_size = 28 * 28
hidden_layer_size = 300
output_layer_size = 10

# 数据加载器父类
class Loader(object):
    def __init__(self, path, count):
        '''
        初始化加载器
        path  数据文件路径
        count 文件中的样本个数
        '''
        self.path = path
        self.count = count

    def get_file_content(self):
        '''
        读取文件内容
        '''
        content = ""
        with open(self.path, "rb") as f:
            content = f.read()
        return content

    def to_int(self, byte):
        '''
        将unsigned byte字符 转为整数 B表示是C语言中的unsigned byte python中的int
        '''
        return struct.unpack('B', byte)[0]

# 图像数据加载器
class ImageLoader(Loader):
    def get_picture(self, content, index):
        '''
        内部函数 从文件中获得图像
        '''
        # 从下标16开始 第一张图16 --> 16 + 28*28 第二张图 16 + 28*28 --> 16 + 28*28+28*28
        start = 16 + index * 28 * 28
        picture = list()
        # 将二进制28 × 28一一转成int 得到每张picture
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(
                    # self.to_int(content[start + i * 28 + j])
                    # 字节流对象的索引结果自动转成int了
                    content[start + i * 28 + j]
                )
        return picture

    def get_one_sample(self, picture):
        '''
        内部函数  将图像转化为样本的输入向量
        '''
        sample = list()
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    def load(self):
        '''
        加载数据文件 获得全部样本的输入向量
        '''
        content = self.get_file_content()
        data_set = list()
        for index in range(self.count):
            data_set.append(self.get_one_sample(self.get_picture(content, index)))
        return data_set

# 标签数据加载器
class LabelLoader(Loader):
    def load(self):
        '''
        加载数据文件 获得全部样本的标签向量
        '''
        content = self.get_file_content()
        labels = list()
        for index in range(self.count):
            labels.append(self.norm(content[8 + index]))
        return labels

    def norm(self, label):
        '''
        内部函数  将一个值转换为10维标签向量
        '''
        label_vector = list()
        label_value = label
        for i in range(10):
            if i == label_value:
                label_vector.append(0.9)
            else:
                label_vector.append(0.1)
        return label_vector


def get_training_data_set(count):
    '''
    获得训练数据集
    '''
    image_loader = ImageLoader('mnist/train-images-idx3-ubyte', count)
    print('训练集样本导入成功...')
    label_loader = LabelLoader('mnist/train-labels-idx1-ubyte', count)
    print('训练集标签导入成功...')
    return (image_loader.load(), label_loader.load())

def get_test_data_set(count):
    '''
    获得测试数据集
    '''
    image_loader = ImageLoader('mnist/t10k-images-idx3-ubyte', count)
    print('测试集样本导入成功...')
    label_loader = LabelLoader('mnist/t10k-labels-idx1-ubyte', count)
    print('测试集标签导入成功...')
    return (image_loader.load(), label_loader.load())

# 输出是10维向量 每一个值代表一个类别的预测值 取最大的值就是预测结果
def get_result(vector):
    max_value_index = 0
    max_value = 0
    for i in range(len(vector)):
        if vector[i] > max_value:
            max_value = vector[i]
            max_value_index = i
    return max_value_index

# 错误率对网络进行评估
def evaluate(neural_network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)

    # 遍历10000次  把每个标签和预测出来的结果 进行比较
    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(neural_network.predict(test_data_set[i]))
        if label != predict:
            error += 1
        return error / total

# 训练策略 每训练10轮  评估一次准确率  当准确率开始下降时终止训练
def train_and_evaluate(neural_network, training_data_count, test_data_count):
    # 上次错误率 1 迭代次数 0  学习率n 0.3
    last_error_ratio = 1.0
    epoch = 0
    learning_rate = 0.3
    train_data_set, train_labels = get_training_data_set(training_data_count)
    test_data_set, test_labels = get_test_data_set(test_data_count)
    # neural_network.dump()
    while True:
        epoch += 1
        neural_network.train(train_data_set, train_labels, 1, learning_rate)
        now = datetime.now()
        now = now.strftime('%Y-%m-%d %H:%M:%S')
        print('{} epoch {} finished'.format(now, epoch))
        if epoch % 10 == 0:
            error_ratio = evaluate(neural_network, test_data_set, test_labels)
            now = datetime.now()
            now = now.strftime('%Y-%m-%d %H:%M:%S')
            print('{} after epoch {}, error ratio is %{}'.format(now, epoch, error_ratio * 100))
            # 如果这次错误率 高于上次 证明过拟合了 结束训练
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio

if __name__ == '__main__':
    neural_network = NeuralNetwork([input_layer_size, hidden_layer_size, output_layer_size])
    training_data_count = 2000
    test_data_count = 1000
    train_and_evaluate(neural_network, training_data_count, test_data_count)
