'''
神经元和感知器不一样的是 感知器激活函数是阶跃函数  神经元的激活函数一般选择sigmoid(0-1)或tanh函数
全连接神经网络就是 最左边的层是输入层(向量x) 最右边的层是输出层(向量y) 中间层是隐藏层
神经元同一层没有连接  第N层的每个神经元都与第N-1层的所有神经元连接(全连接的含义) 上一层输出就是下一层的输入
每个连接都有对应权重w
(输入层)x的特征[工作年份 职位 公司]   中间层       输出层   f激活函数是sigmoid(0-1)
x1(工作年份)                          a4         y1(8)
x2(职位)                              a5         y2(9)
x3(公司)                              a6
                                     a7
中间层的值 a4 = f(w41*x1 +w42*x2 + w43*x3 + b(w44*1))
         a5 = f(w51*x1 +w52*x2 + w53*x3 + b(w54*1))
         a6 = f(w61*x1 +w62*x2 + w63*x3 + b(w64*1))
         a7 = f(w71*x1 +w72*x2 + w73*x3 + b(w7*1))
输出层的值 y1 = f(w84*a4 + w85*a5 + w86*a6 + w87*a7 + w8*1)
          y2 = f(w94*a4 + w95*a5 + w96*a6 + w97*a7 + w9*1)
向量相乘 点积可以一次性算完这些式子  4*3 和 3*1 得到中间层4*1  然后2*4 和 4*1 得到输出2*1
[w41 w42 w43]    [x1]   [a4]        [w84 w85 w86 w87]   [a4]    [y1]
[w51 w52 w53]    [x2]   [a5]        [w94 w95 w96 w97]   [a5]    [y2]
[w61 w62 w63]    [x3]   [a6]                            [a6]
[w71 w72 w73]           [a7]                            [a7]
反向传播算法  要计算本节点的误差项得知道下一层的误差项 所以求误差项的计算顺序是从输出层开始的
然后反向依次计算每个隐藏层的误差项 直到与输入层相连的误差项 当所有节点的误差项计算好后 可以根据式来更新所有的权重
'''

from functools import reduce
import random
import numpy as np
            # NetWork :  神经网络对象
        # Layer    Connections
        # Node     Connection
# 激活函数
def sigmoid(x):
    # exp返回e的x次方  e=2.7
    # return 1.0 / (1 + np.exp(-x))
    # 优化一下 避免出现极大的数据溢出
    if x >= 0:
        return 1.0 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))

# 节点类 负责记录和维护节点自身信息以及与这个节点相关的上下游连接 实现输出值和误差项的计算
class Node(object):
    def __init__(self, layer_index, node_index):
        '''
        layer_index 节点所属的层的编号
        node_index  节点的编号
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = list()
        self.upstream = list()
        self.output = 0
        self.delta = 0

    def set_output(self, output):
        '''
        设置节点的输出值  如果节点属于输入层会用到
        '''
        self.output = output

    def append_downstream_connection(self, connect):
        '''
        添加一个到下游节点的连接
        '''
        self.downstream.append(connect)

    def append_upstream_connection(self, connect):
        '''
        添加一个到上游节点的连接
        '''
        self.upstream.append(connect)

    def calc_output(self):
        '''
        计算节点的输出  这里是输入层之后的层
        比如第二层的节点a4 它的输出为上流节点的输出x1 x2 x3 1 依次乘w41 w42 w43 w44结果相加
        '''
        output = reduce(lambda result, connect: result + connect.upstream_node.output * connect.weight,
                        self.upstream, 0)
        self.output = sigmoid(output)

    def calc_hidden_layer_delta(self):
        '''
        节点是隐藏层时 计算delta
        比如隐藏层第一个节点a4  a4(1-a4)(w84*节点8的误差值, w94*节点9的误差值) 这里的节点8 9也就是输出y1 y2
        '''
        downstream_delta = reduce(lambda result, connect: result + connect.downstream_node.delta *
                                  connect.weight, self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def calc_output_layer_delta(self, label):
        '''
        节点是输出层时  计算delta  输出节点的误差项delta = predict_y(1-predict_y)(y-predict_y)
        '''
        self.delta = self.output * (1 - self.output) * (label - self.output)

    def __str__(self):
        '''
        打印节点的信息
        '''
        node_str = "{0}-{1}: output: {2} delta: {3}".format(self.layer_index,self.node_index,
                                                            self.output,self.delta)
        downstream_str = reduce(lambda result, connect: result + '\n\t' + str(connect), self.downstream,"")
        upstream_str = reduce(lambda result, connect: result + '\n\t' + str(connect), self.upstream, "")
        return node_str + '\n\tdownstream: ' + downstream_str + '\n\tupstream: ' + upstream_str

# 为了实现一个输出恒为1的节点(计算偏置项wb时用)
class ConstNode(object):
    def __init__(self, layer_index, node_index):
        '''
        构造节点对象 输出恒为1  作为每层最后一个节点
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = list()
        self.output = 1

    def append_downstream_connection(self, connect):
        '''
        添加一个到下游节点的连接 b这个节点的连接只有下游的节点
        '''
        self.downstream.append(connect)

    def calc_hidden_layer_delta(self):
        '''
        节点是隐藏层时 计算delta
        比如隐藏层第一个节点a4  a4(1-a4)(w84*节点8的误差值, w94*节点9的误差值) 这里的节点8 9也就是输出y1 y2
        '''
        downstream_delta = reduce(lambda result, connect: result + connect.downstream_node.delta
                                  * connect.weight, self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        node_str = "{}-{}: output: 1".format(self.layer_index, self.node_index)
        downstream_str = reduce(lambda result, connect: result + '\n\t' + str(connect), self.downstream, "")
        return node_str + '\n\tdownstream: ' + downstream_str

class Layer(object):
    def __init__(self, layer_index, node_count):
        '''
        初始化一层
        layer_index  层编号
        node_count  层所包含的节点个数
        '''
        self.layer_index = layer_index
        self.nodes = list()
        # 添加节点(神经元)
        for i in range(node_count):
            self.nodes.append(Node(layer_index, i))
        # 每层最后一个节点为输出恒为1的节点
        self.nodes.append(ConstNode(layer_index, node_count))

    def set_output(self, data):
        '''
        设置层的输出 当层是输入时会用到
        '''
        for i in range(len(data)):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        '''
        计算层的输出向量 输入层后面的层用到
        '''
        for node in self.nodes[:-1]:
            node.calc_output()

    def dump(self):
        '''
        打印层的信息
        '''
        for node in self.nodes:
            print(node)

# 主要记录连接的权重 以及这个连接所关联的上下游节点
class Connection(object):
    def __init__(self, upstream_node, downstream_node):
        '''
        初始化连接 权重初始化为一个小的随机数
        upstream_node 连接的上游节点
        downstream_node
        weight 随机初始化权重
        gradient  梯度  =  上游节点的输出 * 下游节点的误差项
        '''
        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1, 0.1)
        self.gradient = 0.0

    def calc_gradient(self):
        '''
        计算梯度  梯度 = 上游节点的输出(x) * 下游节点的误差项(delta)
        '''
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def update_weight(self, rate):
        '''
        根据梯度下降算法 更新权重  w = w + n*delta*x  梯度(delta是下游的误差项,x是本节点的输出)
        '''
        self.calc_gradient()
        self.weight += rate * self.gradient

    def get_gradient(self):
        return self.gradient

    def __str__(self):
        '''
        打印连接信息
        '''
        return "({}-{})-->({}-{})={}".format(self.upstream_node.layer_index,
                                             self.upstream_node.node_index,
                                             self.downstream_node.layer_index,
                                             self.downstream_node.node_index,
                                             self.weight)

# 提供Connection的集合操作
class Connections(object):
    def __init__(self):
        self.connections = list()

    def add_connection(self, connection):
        self.connections.append(connection)

    def dump(self):
        for connect in self.connections:
            print(connect)

# 神经网络对象  提供API
class NeuralNetwork(object):
    def __init__(self, layers):
        '''
        初始化一个全连接神经网络FCNN
        layers: 二维数组  描述神经网络每层节点数
        初始化每层
        初始化每个神经元之间的连接
        '''
        self.connections = Connections()
        self.layers = list()
        layer_count = len(layers)
        node_count = 0
        # 每层都加入列表
        for i in range(layer_count):
            self.layers.append(Layer(i, layers[i]))
        # 列表生成器 两重循环  让上流的所有神经元都跟下流的每一个神经元建立连接Connection
        for layer in range(layer_count - 1):
            #构造连接时 上游包括constNode都会连接下游不包括constNode的节点 所有constNode只有下游节点
            connections = [Connection(upstream_node, downstream_node)
                           for upstream_node in self.layers[layer].nodes
                           for downstream_node in self.layers[layer + 1].nodes[:-1]]
            for connection in connections:
                connection.downstream_node.append_upstream_connection(connection)
                connection.upstream_node.append_downstream_connection(connection)
                self.connections.add_connection(connection)

    def train(self, data_set, labels, iteration, rate):
        '''
        训练神经网络
        labels: 数组 训练样本标签  每个元素是一个样本的标签(真实值y)
        data_set: 二维数组 训练样本特征  每个元素是一个样本的特征
        '''
        for i in range(iteration):
            for input_feature in range(len(data_set)):
                self.train_one_sample(data_set[input_feature], labels[input_feature], rate)

    def train_one_sample(self, sample, label, rate):
        '''
        内部函数  用一个样本训练网络
        predict 计算神经网络每层每个神经元的输出
        calc_delta  计算每个神经元的误差项 反向传播 从输出层开始往前计算
        update_weight 根据每个节点的误差项 计算出每条连接的梯度 根据梯度下降算法来更新权重
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def calc_delta(self, label):
        '''
        内部函数  计算每个节点的delta  反向传播 先计算输出层的神经元的误差项 再往前推
        '''
        output_nodes = self.layers[-1].nodes
        # 输出层误差项
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        # 从输出层的前一层一直到输入层反向计算误差
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        '''
        内部函数  更新每个连接权重
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for connect in node.downstream:
                    connect.update_weight(rate)

    def calc_gradient(self):
        '''
        内部函数  计算每个连接的梯度
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for connect in node.downstream:
                    connect.calc_gradient()

    def get_gradient(self, sample, label):
        '''
        获得网络在一个样本下 每个连接上的梯度
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()

    def predict(self, sample):
        '''
        根据输入的样本预测输出值
        sample: 数组 样本的特征 也就是网络的输入向量
        返回输出层每个节点的输出
        '''
        # 给输入层设置输出值
        self.layers[0].set_output(sample)
        # 剩下的层计算输出
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        return list(map(lambda node: node.output, self.layers[-1].nodes[:-1]))

    def dump(self):
        '''
        打印网络信息
        '''
        for layer in self.layers:
            layer.dump()

class Normalizer(object):
    def __init__(self):
        # 16进制 1 2 4 8 16 32 64
        self.mask = [
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
            # 0x1, 0x2, 0x4
        ]

    # 将值归一化到0-1
    def norm(self, number):
        # 按位与运算 要是每个位都不相同 得到0 也就是0.1 否则就是0.9
        return list(map(lambda mask: 0.9 if number & mask else 0.1, self.mask))

    # 将0-1的值 反归一化回来
    def denorm(self, vec):
        # 二值化  最后得到的结果 大于0,5就为1 小于就为0
        binary = list(map(lambda i: 1 if i > 0.5 else 0, vec))
        for i in range(len(self.mask)):
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda x, y: x + y, binary)

# 构造数据集
def train_data_set():
    normalizer = Normalizer()
    data_set = list()
    labels = list()
    for i in range(0, 256, 8):
        n = normalizer.norm(int(random.uniform(0, 256)))
        data_set.append(n)
        labels.append(n)
    return (data_set, labels)

# 训练
def train(neural_network):
    data_set, labels = train_data_set()
    learning_rate = 0.5
    neural_network.train(data_set, labels, 50, learning_rate)

# 均方误差
def mean_square_error(vec1, vec2):
    return lambda vec1, vec2: 0.5 * \
                       reduce(lambda a, b: a + b,
                              list(map(lambda v1, v2: (v1 - v2) ** 2, vec1, vec2)))

# 测试一下神经网络有没有问题 或者算法本身有没有问题 用梯度检查方法
# 1. 使用一个样本d对神经网络进行训练 获得每个权重的梯度
# 2. 将w加上一个很小的值 重新计算神经网络在这个样本d下的Ed+
# 3. 将w减去一个很小的值 重新计算神经网络在这个样本d下的Ed-
# 4. 计算出期望的梯度值  和第一步获得的梯度值比较 应该几乎相等
def gradient_check(neural_network, sample_feature, sample_label):
    '''
    梯度检查
    neural_network  神经网络对象
    sample_feature  样本特征x
    sample_label    样本标签y
    '''
    # 获得网络在当前样本下每个连接的梯度
    neural_network.get_gradient(sample_feature, sample_label)

    # 对每个权重做梯度检查
    for connection in neural_network.connections.connections:
        # 获取指定连接的梯度
        actual_gradient = connection.get_gradient()
        # 增加一个很小的值  计算网络的误差
        epsilon = 0.0001
        connection.weight += epsilon
        # 用输出层每个节点的输出 和 真实值y 计算均方误差
        error1 = mean_square_error(neural_network.predict(sample_feature), sample_label)

        # 减去一个很小的值  计算均方误差
        connection.weight -= (epsilon * 2)
        error2 = mean_square_error(neural_network.predict(sample_feature), sample_label)

        # 计算期望的梯度值 f(w + △) - f(w - △) / 2△
        expected_gradient = (error2 - error1) / (2 * epsilon)

        # 打印
        print('expected gradient: {}\t\nactural gradient: \t{}'.format(expected_gradient,
                                                                       actual_gradient))
def gradient_check_test():
    neural_network = NeuralNetwork([2, 2, 2])
    sample_feature = [0.9, 0.1]
    sample_label = [0.9, 0.1]
    gradient_check(neural_network, sample_feature, sample_label)

def test(neural_network, data):
    normalizer = Normalizer()
    # 归一化数据
    normalizer_data = normalizer.norm(data)
    # 预测的结果
    predict_data = neural_network.predict(normalizer_data)
    print('test_data: {}\tpredict: {}'.format(data, normalizer.denorm(predict_data)))


# 正确的比率
def correct_ratio(neural_network):
    normalizer = Normalizer()
    correct = 0.0

    # 256个值 每个拿去mask成0.1 0.9的列表 然后拿去推理 得到预测值再反归一化回来 看看是否和真实值相等
    for i in range(256):
        if normalizer.denorm(neural_network.predict(normalizer.norm(i))) == i:
            correct += 1.0
    print('model correct ratio: %{}'.format(correct / 256 * 100))


if __name__ == '__main__':
    # 输入层8个节点 隐藏层3个节点 输出层8个节点
    neural_network = NeuralNetwork([8, 10, 8])
    # 训练
    train(neural_network)
    # 打印神经网络里 每层 每个神经元的信息
    neural_network.dump()
    # gradient_check_test()
    # test(neural_network, 50)
    correct_ratio(neural_network)