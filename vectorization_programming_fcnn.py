'''
现在告别面向对象编程  使用更适合学习算法的编程方式： 向量化编程
1. 因为不用真的写Node Layer那样的类 只要实现算数计算就行了
2. 底层算法库会针对向量运算做优化   程序效率会提升很多  我们把计算表达为向量的形式
'''
import numpy as np
from functools import reduce

# 全连接层实现类
class FullConnectedLayer(object):
    def __init__(self, input_size, output_size, activator):
        '''
        构造函数
        input_size  本层输入向量的维度
        output_size 本层输出向量的维度
        activator   激活函数
        '''
        self.input_size = input_size
        self.output_size = output_size
        self.activator = activator
        # 权重数组w  第一层到第二层8 10  那么权重数组维度就是10 * 8(10行8列)
        self.w = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        # 偏置项b 第一层 到 第二层8 10  偏置数组维度就是10 * 1
        self.b = np.zeros((output_size, 1))
        # 输出向量 维度就是10 * 1
        self.output = np.zeros((output_size, 1))

    def forward(self, input_array):
        '''
        前向计算  a = f(w .* x)
        input_array   输入向量 维度必须等于input_size
        将两层之间的w*x+b全部计算完 得到下流层的所有节点输出值
        '''
        self.input = input_array
        self.output = self.activator.forward(
            # np.dot 矩阵相乘 (10 * 8) .* (8 * 1) = 10 * 1
            np.dot(self.w, input_array) + self.b
        )

    def backward(self, delta_array):
        '''
        反向计算
        delta_array 从上一层传递过来的误差项
        np.array的.T表示它的转置矩阵 x是本层的节点的输出值
        本层(中间层)的误差项 = x * (1-x) * wT .* delta_array
        w的梯度是上一层误差项 delta_array .* xT
        b的梯度就是上一层的误差项 delta_array
        '''
        self.delta = self.activator.backward(self.input) * np.dot(
            self.w.T, delta_array)
        self.w_gradient = np.dot(delta_array, self.input.T)
        self.b_gradient = delta_array

    def update(self, learning_rate):
        '''
        使用梯度下降算法更新权重  括号括起来的是w b的梯度
        w = w + n * (上层的误差项delta .* 本层的输入的转置矩阵xT)
        b = b + n * (上层的误差项delta)
        '''
        self.w += learning_rate * self.w_gradient
        self.b += learning_rate * self.b_gradient

    def dump(self):
        print('W: {}{}\nb: {}{}'.format(self.w.shape, self.w, self.b.shape, self.b))


# Sigmoid激活函数类
class SigmoidActivator(object):
    def forward(self, weighted_input):
        '''
        前向计算的激活函数
        '''
        # return list(map(lambda x: 1.0 / (1.0 + np.exp(-x)) if x >= 0
        #             else np.exp(x) / (1.0 + np.exp(x)), weighted_input))
        return 1.0 / (1.0 + np.exp(-weighted_input))

    def backward(self, output):
        '''
        反向计算的激活函数
        '''
        return output * (1 - output)

# 神经网络类
class NeuralNetwork(object):
    def __init__(self, layers):
        '''
        构造函数
        layers  列表 里面有多少值代表有多少层 每个值代表每层的节点数
        有N层 就添加N-1个全连接层对象 到神经网络
        '''
        self.layers = list()
        for i in range(len(layers) - 1):
            self.layers.append(
                FullConnectedLayer(layers[i], layers[i+1], SigmoidActivator())
            )

    def train(self, data_set, labels, epoch, rate):
        '''
        训练函数
        data_set 输入样本  256个np.array(8维列向量)
        labels: 样本标签   256个np.array(8维列向量)
        epoch  训练轮数
        rate   学习速率
        '''
        for i in range(epoch):
            for d in range(len(data_set)):
                self.train_one_sample(data_set[d], labels[d], rate)

    def train_one_sample(self, sample, label, rate):
        '''
        内部函数  用一个样本训练网络
        sample  输入特征x  8维列向量
        labels  标签y      8维列向量
        predict     前向计算 计算神经网络每层每个神经元的输出
        calc_delta  反向计算 从输出层开始往前计算 每个神经元的误差项 利用上层的误差项得到本层的误差项和w,b梯度
        update_weight 有了w, b的梯度 根据梯度下降算法来更新权重(往梯度反方向改变w b的值)
        '''
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)

    def predict(self, sample):
        '''
        使用神经网络实现预测  计算所有节点的输出值
        sample   输入样本 8维列向量
        '''
        output = sample
        for layer in self.layers:
            # 全连接层对象调用前向计算 每次得到下一层的输出传入作为本次的输入 最后返回结果
            layer.forward(output)
            output = layer.output
        return output

    def calc_gradient(self, label):
        '''
        节点是输出层时  输出节点的误差项delta = output(1-output)(label-output)
        有了输出层的delta 从输出层反向计算 依次得到前面层的误差项 以及该层w和b的梯度 更新权重时使用
        '''
        delta = self.layers[-1].activator.backward(self.layers[-1].output) * \
                                                  (label - self.layers[-1].output)

        for layer in self.layers[::-1]:
            '''
            遍历一层 就是根据上层误差项反向计算出这层的误差项 和w b的梯度（更新权重使用）
            再用这一层得到的误差项 继续求下一层的误差项 w b的梯度
            '''
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        '''
        遍历N-1个连接层 更新每个权重  w += n * w_gradient b += n * b_gradient
        '''
        for layer in self.layers:
            layer.update(rate)

    def dump(self):
        for layer in self.layers:
            layer.dump()

    def loss(self, output, label):
        '''
        代价函数 损失函数 目标函数  求均方误差MSE
        '''
        return 0.5 * ((label - output)**2).sum()

    def gradient_check(self, sample_feature, sample_label):
        '''
        梯度检查
        '''
        # 获得网络在当前样本下每个连接的梯度
        self.predict(sample_feature)
        self.calc_gradient(sample_label)

        # 检查梯度  1 * 10-4次方
        epsilon = 1e-4
        # 遍历全连接层的w数组 每个值都加减一个epsilon 来看均方误差
        for layer in self.layers:
            for i in range(layer.w.shape[0]):
                for j in range(layer.w.shape[1]):
                    layer.w[i, j] += epsilon
                    output = self.predict(sample_feature)
                    error1 = self.loss(output, sample_label)

                    layer.w[i, j] -= 2 * epsilon
                    output = self.predict(sample_feature)
                    error2 = self.loss(output, sample_label)
                    expect_gradient = (error2 - error1) / (2 * epsilon)
                    layer.w[i, j] += epsilon
                    print('weights({}{}): expected-actural: {} : {}'.format(i, j, expect_gradient, layer.w_gradient[i, j]))

class Normalizer(object):
    def __init__(self):
        self.mask = [
            0x1, 0x2, 0x4, 0x8, 0x10, 0x20, 0x40, 0x80
        ]

    # 位运算 输入为8维向量  0-255的值的8位二进制从低位到高位 位是1就是0.9 位是0就是0.1
    def norm(self, number):
        data = list(map(lambda mask: 0.9 if number & mask else 0.1, self.mask))
        # 这里从1行8列的行向量 变成8行1列的列向量
        return np.array(data).reshape(8, 1)

    def denorm(self,vector):
        # 二值化 np.array是 [:::] 行x:x 列x:x 步长x  这里取了所有行的第1列 就变回了行向量
        binary = list(map(lambda i: 1 if i > 0.5 else 0, vector[:, 0]))
        for i in range(len(self.mask)):
            # 还原8位二进制 为对应数值
            binary[i] = binary[i] * self.mask[i]
        return reduce(lambda a, b: a + b, binary)

def gradient_check():
    '''
    梯度检查
    '''
    data_set, labels = transpose(train_data_set())
    neural_network = NeuralNetwork([8, 10, 8])
    neural_network.gradient_check(data_set[0], labels[0])

# 转置矩阵
def transpose(args):
    '''
    args是元祖 有两个列表
    第一个是数据集列表 里面装有256个np.array 每个np.array是一个8行1列的列向量
    第一个是标签列表   里面装有256个np.array 每个np.array是一个8行1列的列向量
    '''
    return list(map(lambda arg:
                list(map(lambda line: np.array(line).reshape(len(line), 1), arg)),
                     args))

def train_data_set():
    normalizer = Normalizer()
    data_set = list()
    labels = list()
    for i in range(0, 256):
        n = normalizer.norm(i)
        data_set.append(n)
        labels.append(n)
    return data_set, labels

def correct_ratio(neural_network):
    '''
    用正确率来评估网络
    '''
    normalizer = Normalizer()
    correct = 0.0
    for i in range(256):

        if normalizer.denorm(neural_network.predict(normalizer.norm(i))) == i:
            correct += 1

    print('correct ratio: %{}'.format(correct / 256 * 100))

def test():
    # 转置将数据集和标签转成列向量
    data_set, labels = train_data_set()
    neural_network = NeuralNetwork([8, 10, 8])
    rate = 0.5
    mini_batch = 20
    epoch = 10
    # neural_network.dump()
    for i in range(epoch):
        neural_network.train(data_set, labels, mini_batch, rate)
        print('after epoch {} loss: {}'.format(i + 1,
                neural_network.loss(neural_network.predict(data_set[-1]), labels[-1])))
        rate /= 2
    correct_ratio(neural_network)

if __name__ == '__main__':
    test()
    # gradient_check()