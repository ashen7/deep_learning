#有些数据集不是线性可分的 这样感知器可能无法收敛  使用一个线性函数替代感知器的阶跃函数 这种感知器叫线性单元
# 线性单元在面对线性不可分的数据集时 会收敛到一个最佳的近似上   解决回归问题
# 模型的输入x(work_time,trade,position,company)多维向量 其中每一个表示一个特征 对应不同的权重w  来预测y收入
# 监督学习 就是输入特征x 对应输出y(真实值) 让模型找到规律自己可以预测没见过的值(泛化)
# 无监督学习 只有输入特征x 没有输出y 比如语音转文本 带标签的文本太少 先用无监督学习将一些相似音节聚类 在用少量带标签y训练
# 梯度是一个向量 它指向函数值上升最快的方向 所以梯度下降优化算法就是沿梯度反方向走（幅度是学习率） 直到函数最小值附近(一阶导为0)

#代码复用    线性单元(随机梯度下降)和感知器只有激活函数是不同的
from perceptron import Perceptron
import matplotlib.pyplot as plt

# 定义激活函数
f = lambda x:x

# 线性单元
class LinearUnit(Perceptron):
    def __init__(self, input_num):
        # 调用父类构造
        Perceptron.__init__(self, input_num, f)

def get_training_dataset():
    # 构造5个人的收入数据  只有1个特征 工作年限
    input_vector = [[5],[3],[8],[1.4],[10.1]]
    labels = [5500, 2300, 7600, 1800, 11400]
    return (input_vector, labels)

def train_linear_unit():
    linear_unit = LinearUnit(1)
    # 训练 迭代10轮 学习率0.01
    input_vector, labels = get_training_dataset()
    linear_unit.train(input_vector, labels, 10, 0.01)
    return linear_unit

def plot(linear_unit):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # ax.scatter(map(lambda x:x[0], input_vector), labels)

    # 权重 偏置
    weights = linear_unit.weights
    bias = linear_unit.bias
    # x是工作年份 y是训练得到的最优权重*x 加最优偏置
    x = range(0, 12, 1)
    y = list(map(lambda x: weights[0] * x + bias, x))
    ax.plot(x, y)
    plt.show()

if __name__ == '__main__':
    linear_unit = train_linear_unit()
    print(linear_unit)
    print("Work 3.4 years, monthly salary={}".format(linear_unit.predict([3.4])))
    print("Work 15 years, monthly salary={}".format(linear_unit.predict([15])))
    print("Work 1.5 years, monthly salary={}".format(linear_unit.predict([1.5])))
    print("Work 6.3 years, monthly salary={}".format(linear_unit.predict([6.3])))
    plot(linear_unit)