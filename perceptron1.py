#感知器 就是神经元
#w是与输入对应的权重 b是偏置(输入x值永远为1对应的权重) t是训练样本的值 一般称之为label
#y是输出 n是学习率 作用是控制每一步调整权重的幅度
#输入x1 x2 x3 对应权重w1 w2 w3 给到神经元 也就是w1x1+w2x2+w3x3+b 最后送给激活函数得到输出
from functools import reduce

class Perceptron(object):
    def __init__(self, input_num, activator):
        # 初始化感知器 设置输入参数个数 和激活函数
        # 激活函数的类型
        self.activator = activator
        # 权重初始化为0
        self.weights = [0.0 for i in range(input_num)]
        # 偏置初始化为0
        self.bias = 0.0

    def __str__(self):
        "打印学习到的权重 偏置"
        return "weights:{0}\tbias:{1}".format(self.weights, self.bias)

    def predict(self, input_vector):
        # 输入向量 输出感知器的计算结果 recude接收一个函数 和一个列表\元祖 依次取两个参数调用函数
        # map函数接收一个函数 一个列表\元祖 取每个元素调用函数
        # return self.activator(reduce(lambda a, b:a + b,
        #                          list(map(lambda x, w:x * w,
        #                          input_vector, self.weights))) + self.bias)
        wx = list(map(lambda x, w:x * w, input_vector, self.weights))
        predict_output = reduce(lambda a,b:a+b,wx) + self.bias
        print("预测值：", predict_output)
        predict_output = self.activator(predict_output)
        print("经过激活函数后的预测值：", predict_output)
        return predict_output

    def train(self, input_vector, labels, iteration, rate):
        # 输入训练数据：一组向量x 与每个向量对应的label(期望输出)也是真实值y 以及训练迭代轮数 学习率
        for i in range(iteration):
            print("\n==============================这是第{}轮训练===========================".format(i))
            self.one_iterator(input_vector, labels, rate)

    def one_iterator(self, input_vector, labels, rate):
        # 一次迭代 把所有的训练数据过一遍
        # 把输入和输出打包在一起 成为样本的列表(input_vector,label) 也就是 x 和 y
        samples = zip(input_vector, labels)
        for (input_vector, label) in samples:
            # 计算感知器在当前权重下的输出
            print("输入: ",input_vector, "真实值： ",label)
            predict_output = self.predict(input_vector)
            # 更新权重
            self.update_weights(input_vector, predict_output, label, rate)

    def update_weights(self, input_vector, predict_output, label, rate):
        # 按照感知器规则更新权重  delta是△变化量  等于真实值 - 预测的输出值
        delta = label - predict_output
        print("真实值：",label,"预测值:",predict_output,"变化量△ 真实值减去预测值:",delta)
        #w = w + △w  而△w = n(t - y)x t是样本值 y是真实值
        self.weights = list(map(lambda x,w:w + rate * delta * x,
                                input_vector, self.weights))
        print("更新后的权重：", self.weights)
        # 更新bias b = b + △b 而△b = n(t - y)
        self.bias += rate * delta
        print("更新后的偏置：", self.bias,"\n")
# 激活函数 这里激活函数用阶跃函数 大于0为1 其他为0
def f(x):
    return 1 if x > 0 else 0

# 基于and真值表构建训练数据
def get_training_dataset():
    # 输入向量列表x
    input_vector = [[0,0],[0,1],[1,0],[1,1]]
    # (期望)真实的输出列表y
    labels = [0, 0, 0, 1]
    return (input_vector, labels)

# 使用and真值表训练感知器
def train_and_perceptron():
    # 创建感知器 输入参数个数为2 因为and是二元函数 x1 x2 激活函数为f
    perceptron = Perceptron(2, f)
    #训练 迭代10轮 学习速率为0.1
    input_vector, labels = get_training_dataset()
    perceptron.train(input_vector, labels, 1000, 0.1)
    #返回感知器对象
    return perceptron

if __name__ == '__main__':
    # 训练与运算and的感知器
    and_perception = train_and_perceptron()
    # 打印训练获得的权重
    print(and_perception)
    # 测试
    print("0 and 0 = {}".format(and_perception.predict([0, 0])))
    print("0 and 1 = {}".format(and_perception.predict([0, 1])))
    print("1 and 0 = {}".format(and_perception.predict([1, 0])))
    print("1 and 1 = {}".format(and_perception.predict([1, 1])))
