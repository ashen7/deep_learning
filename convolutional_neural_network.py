'''
cnn卷积神经网络 vs  fcnn全连接神经网络
1. 参数数量太多:  一张图片1000*1000像素 比如第一个隐藏层节点为100 光第一层就有（1000*1000 + 1）×100=1一亿个参数
2. 没有利用像素之间的位置信息:  对于图像识别任务来说 每个像素和其周围像素联系密切 和离的远的像素联系可能就很小了
如果一个神经元和上一层所有神经元相连 那么就相当于把图像的所有像素都等同看待 有大量的权重很小(这些连接没有用)
3. 网络层次限制:  网络层数越多其表达能力越强 但是全连接神经网络的梯度很难传递超过3层

cnn有3个主要思路  解决了这些问题 尽可能保留重要参数 去掉大量不重要的参数 达到更好的学习效果
1. 局部连接： 每个神经元只和上一层一小部分神经元相连
2. 权值共享:  一组连接可以共享一个权重 而不是每个连接都有一个不同的权重
3. 下采样：  使用Pooling来减少每层的样本数

cnn的激活函数 使用ReLu函数   f(x) = max(0, x) 值大于0等于本身 小于0就等于0
RuLu比起sigmoid 优点是速度快 减轻梯度消失问题 所以使用ReLu可以训练更深的网络

cnn的网络架构： input -> [conv*N -> pool]*M -> fc*K
一个卷积神经网络右若干卷积层 Pooling池化层 全连接层组成
三维的层结构： 卷积神经网络每层的神经元是安装三维排列的
1. 输入层的宽高对应于输入图像的宽和高 深度为1
2. 第一个卷积层对这幅图像进行卷积操作 得到3个Feature Map特征图 3是表示这个卷积层有3个Filter
也就是有3套参数  三个Filter对原始图像提取出三组不同的特征 得到三个Feature Map特征图
3. 接着Pooling池化层对这三个Feature Map做下采样得到三个变小的Feature Map
4. 第二个卷积层 有5个Filter 每个Filter都把前面三个Feature Map卷积在一起 得到新的Feature Map 这样得到5个
5. 第二次池化层 下采样得到5个变小的Feature Map
6. 全连接层和上一层的5个Feature Map中的每个神经元相连 输出层再和这一层每个神经元相连 得到网络的输出

卷积的计算： image 5 * 5 ,  filter 3 * 3 就是在5×5的像素值上选3*3 和filter位置对应相乘然后相加
最后ReLu函数得到结果值 作为Feature Map的第一个值 然后3*3在图像往右一格继续计算 得到Feature Map的第二个值
feature_map_width = (image_width - filter_width + 2pad) / stride + 1
feature_map_height = (image_height - filter_height + 2pad) / stride + 1
比如 图像宽5 filter宽3 Zero Padding为0（在原始图像周围补几圈0 这对图像边缘部分的特征提取很有帮助）
stride步长2   得到feature_map为2*2
对于包含两个3*3*3的filter的卷积层来说 其参数数量(w,b)仅有(3*3*3+1)*2=56个 参数数量与上一层神经元个数无关
这就体现了cnn的权重共享和局部连接

卷积神经网络中的卷积 和 数学的卷积有些区别  cnn中的卷积叫(互相关操作cross-correlation)
卷积和互相关操作可以转化  把矩阵A翻转180度 再交换矩阵A和B的位置 那么卷积就变成了互相关操作

池化层Pooling的计算： 主要作用是下采样 去掉Feature Map中不重要的样本
最要方法有Max Pooling（取x*x样本的最大值为该样本值） 和Mean Pooling(取x*x样本的平均值为该样本值)

最后是全连接层 用特征图和神经元相连得到输出



cnn的训练 反向传播算法： 利用链式求导计算损失函数对每个权重的偏导(梯度) 根据梯度下降公式更新权重
1. 前向计算每个神经元的输出值
2. 反向计算每个神经元的误差项delta(也叫敏感度 实际上是网络的损失函数Ed对神经元加权输入net的偏导)
3. 得到误差项delta 来计算每个神经元连接权重的梯度   根据梯度下降算法 来更新每个权重值

卷积层的训练：
1. 前向计算卷积层的输出值
2. 将误差项传递到上一层
    1. （假设本层的误差项delta计算好了）步长为1 输入图片深度为1 filter个数为1最简单的情况
        上一层delta = 本层delta *(卷积操作) W（这个filter翻转180度） * f'(卷积计算的输出值)
    2. 步长为S
        步长为S就是跳过了步长为1到s-1时行列的计算
        对步长为S的sensitivity map相应的位置补0 将其还原成步长为1时的sensitivity map
    3. 输入深度为D
        输入深度为D filter的深度也为D了 因为di通道只与相应di通道卷积 比如input d1与filter d1卷积
        所有用filter的di通道权重 对本层的sensitivity map卷积 得到上一层di通道的sensitivity map
    4. filter数量为N
        filter数量为N 输出层的深度也为N 第i个filter卷积产生输出层的第i个feature map
        由于上一层每个加权输入都同时影响了本层所有特征图的输出值 所以反向计算误差项时 用全导数公式
        先用第d个filter对本层相应第d个sensitivity map卷积 得到一组N个上一层的偏sensitivity map
        依次 会得到d组偏sensitivity map 最后各组之间将N个偏sensitivity map按元素相加
        得到最终N个上一层的sensitivity map
3. 卷积层filter权重梯度的计算
    得到本层sensitivity map 计算filter的权重梯度 由于卷积层是权重共享的
    sensitivity map作为卷积核 在Input上进行互相关操作 就可以得到最终的权重梯度
    而偏置项wb的梯度 就是sensitivity map的所有误差项之和

池化层Pooling层的训练：
无论是max pooling还是mean pooling都没有需要学习的参数
因此 cnn训练中 Pooling层仅仅是将误差项传递到上一层 而没有梯度的计算
max pooling  下一层的误差项的值会原封不动的传递给上一层对应x*x块中最大值对应的神经元 其他神经元误差项为0
mean pooling 下一层的误差项的值会平均分配到上一层对应区块的所有神经元
'''

import numpy as np

# 获取卷积区域 也可以获取池化区域
def get_region(input_array, row, col, filter_width, filter_height, stride):
    '''
    从输入数组中获取本次卷积的区域
    自动适配输入为2D和3D的情况
    '''
    start_row = row * stride
    start_col = col * stride
    if input_array.ndim == 2:
        return input_array[
            start_row:start_row + filter_height,
            start_col:start_col + filter_width
        ]
    elif input_array.ndim == 3:
        return input_array[:,
            start_row:start_row + filter_height,
            start_col:start_col + filter_width
        ]

# 获得一个2d区域内 x*x 中最大值所在索引
def get_max_index(array):
    max_row = 0
    max_col = 0
    max_value = array[0, 0]
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i][j] > max_value:
                max_value = array[i][j]
                max_row = i
                max_col = j
    return (max_row, max_col)

# 计算卷积
def convolution(input_array, kernel_array, feature_map, stride, bias):
    '''
    计算卷积 自动适配输入为2D和3D的情况
    '''
    channel_number = input_array.ndim
    # 这里输出的维度肯定是2维 多个filter计算每个feature map
    feature_map_height = feature_map.shape[0]
    feature_map_width = feature_map.shape[1]
    # 这里不确定是否是2d 如果是3d 则第一个值不是行 而是深度
    kernel_height = kernel_array.shape[-1]
    kernel_width = kernel_array.shape[-2]
    for i in range(feature_map_height):
        for j in range(feature_map_width):
            feature_map[i][j] = (
                get_region(input_array, i, j, kernel_width,
                          kernel_height, stride) * kernel_array).sum() + bias

# 为数组添加Zero Padding
def padding(input_array, zero_padding):
    '''
    自动适配输入为2D和3D的情况
    '''
    if zero_padding == 0:
        return input_array
    else:
        if input_array.ndim == 3:
            input_depth = input_array.shape[0]
            input_height = input_array.shape[1]
            input_width = input_array.shape[2]
            # 构建一个填充后的矩阵
            padded_array = np.zeros((
                input_depth,
                input_height + 2 * zero_padding,
                input_width + 2 * zero_padding
            ))
            # 给矩阵原来的区域赋值
            padded_array[:,
                         zero_padding:zero_padding + input_height,
                         zero_padding:zero_padding + input_width] = input_array
            return padded_array
        elif input_array.ndim == 2:
            input_height = input_array.shape[0]
            input_width = input_array.shape[1]
            # 构建一个填充后的矩阵
            padded_array = np.zeros((
                input_height + 2 * zero_padding,
                input_width + 2 * zero_padding
            ))
            # 给矩阵原来的区域赋值
            padded_array[zero_padding:zero_padding + input_height,
                         zero_padding:zero_padding + input_width] = input_array
            return padded_array

def element_wise_op(array, op):
    '''
    np.nditer 返回可迭代对象
    对numpy数组进行按元素操作 并将返回值写回到数组中
    '''
    for i in np.nditer(array, op_flags=['readwrite']):
        i[...] = op(i)

# 卷积层
class ConvolutionLayer(object):
    def __init__(self, input_width, input_height, channel_number,
                 filter_width, filter_height, filter_number,
                 zero_padding, stride, activator, learning_rate):
        '''
        卷积层构造函数  初始化卷积层的超参数
        输入图像宽，高，深度  卷积核宽，高，深度   补0填充  步长   激活函数   学习率
        输出特征图
        '''
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.filter_number = filter_number
        self.zero_padding = zero_padding
        self.stride = stride
        self.output_width = ConvolutionLayer.calculate_output_size(
            self.input_width, self.filter_width, self.zero_padding, self.stride
        )
        self.output_height = ConvolutionLayer.calculate_output_size(
            self.input_height, self.filter_height, self.zero_padding, self.stride
        )
        # 深度  行  列
        self.output_array = np.zeros((self.filter_number, self.output_height, self.output_width))
        self.filters = list()
        for i in range(self.filter_number):
            self.filters.append(Filter(self.filter_width, self.filter_height, self.channel_number))
        self.activator = activator
        self.learning_rate = learning_rate


    def forward(self, input_array):
        '''
        计算卷积层的输出
        输出结果保存在self.output_array
        '''
        self.input_array = input_array
        self.padded_input_array = padding(input_array, self.zero_padding)
        for i in range(self.filter_number):
            # 遍历filter个数 每一个filter 计算卷积得到一个feature map
            filter = self.filters[i]
            convolution(self.padded_input_array, filter.get_weights(),
                        self.output_array[i], self.stride, filter.get_bias())
        # 将feature map进行ReLu激活函数 得到结果
        element_wise_op(self.output_array, self.activator.forward)

    def backward(self, input_array, sensitivity_array, activator):
        '''
        1. 前向计算特征图的输出
        2. 反向计算传递给前一层的误差项
        3. 利用后一层的误差项计算前一层的每个权重的梯度
            前一层的误差项保存在self.delta_array
            梯度保存在Filter对象的weights_gradient
        '''
        self.forward(input_array)
        self.bp_sensitivity_map(sensitivity_array, activator)
        self.bp_gradient(sensitivity_array)

    def update(self):
        '''
        安装梯度下降算法 更新权重
        '''
        for filter in self.filters:
            filter.update(self.learning_rate)

    def bp_sensitivity_map(self, sensitivity_array, activator):
        '''
        计算传递到上一层的sensitivity map:
            1. 本层的sensitivity map还原到步长为1的sensitivity map
            2. sensitivity map外圈补0
            3. 将每个filter(权重数组翻转180度 遍历每个深度和对应filter的sensitivity map进行卷积)
            4. 这样得到每个filter对应的上一层sensitivity map 把他们加起来的和就是上一层的sensitivity map
        sensitivity_array: 本层的sensitivity map
        activator: 上一层的激活函数
        '''
        # 处理卷积步长 对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        # 对sensitivity map进行zero padding
        expanded_width = expanded_array.shape[2]
        zero_padding = int((self.input_width + self.filter_width - 1 - expanded_width) / 2)
        padded_array = padding(expanded_array, zero_padding)
        # 初始化delta_array  用于保存传递给上一层的sensitivity map
        self.delta_array = self.create_delta_array()
        # 对于具有多个filter的卷积层来说  最终传递到上一层的sensitivity map
        # 相当于是所有的filter的sensitivity map之和
        for f in range(self.filter_number):
            filter = self.filters[f]
            # 将filter权重翻转180度  np.rot90(array, 2)是翻转180度  用map是把w的每个深度都翻转180度
            flipped_weights = np.array(list(map(lambda a: np.rot90(a, 2),
                                                filter.get_weights())))
            # 计算与一个filter对应的delta_array
            delta_array = self.create_delta_array()
            for d in range(delta_array.shape[0]):
                # 深度不为1时 filter个数也不为1时
                # 一个filter的sensitivity map 卷积 翻转过后的这个filter的深度d个权重数组
                # 得到一个filter的上一层误差项
                convolution(padded_array[f], flipped_weights[d],
                            delta_array[d], 1, 0)
            # 这里加上每个filter的上一层误差项 最后保存的就是(filter为多个时)上一层的sensitivity map
            self.delta_array += delta_array

        # 将计算结果与激活函数的偏导 做element-wise乘法操作
        derivative_array = np.array(self.input_array)
        element_wise_op(derivative_array, activator.backward)
        # 最终上一层的sensitivity map就是 delta_array * 激活函数(上一层的加权输入)的导数
        self.delta_array *= derivative_array

    def bp_gradient(self, sensitivity_array):
        '''
        计算每个filter的每个权值w的梯度:
            1. 用sensitivity map作为卷积核  还原成步长为1的sensitivity map
            2. 在input上进行互相关操作 得到的就是filter的权重w的梯度
            3. 偏置项的梯度就是sensitivity map所有误差项之和
        '''
        # 处理卷积步长 对原始sensitivity map进行扩展
        expanded_array = self.expand_sensitivity_map(sensitivity_array)
        for f in range(self.filter_number):
            # 计算每个filter的每个权重的梯度
            filter = self.filters[f]
            for d in range(filter.weights.shape[0]):
                convolution(self.padded_input_array[d], expanded_array[f],
                            filter.weights_gradient[d], 1, 0)
            # 计算每个filter的偏置项的梯度
            filter.bias_gradient = expanded_array[f].sum()

    def expand_sensitivity_map(self, sensitivity_array):
        '''
        将步长为S的sensitivity map 还原成 步长为1的sensitivity map
        '''
        depth = sensitivity_array.shape[0]
        # 确定扩展后sensitivity map的大小  就是计算步长为1时sensitivity map的大小
        expanded_width = (self.input_width - self.filter_width + 2 * self.zero_padding + 1)
        expanded_height = (self.input_height - self.filter_height + 2 * self.zero_padding + 1)
        # 构建新的sensitivity map
        expand_array = np.zeros((depth, expanded_height, expanded_width))
        # 从原始sensitivity map把误差值赋值过来相应地方
        for i in range(self.output_height):
            for j in range(self.output_width):
                row_position = i * self.stride
                col_position = j * self.stride
                expand_array[:, row_position, col_position] = sensitivity_array[:, i, j]
        return expand_array

    def create_delta_array(self):
        '''
        创建用来保存传递给上一层的sensitivity map的数组
        '''
        return np.zeros((self.channel_number, self.input_height, self.input_width))

    @staticmethod
    def calculate_output_size(input_size, filter_size, zero_padding, stride):
        # feature_map_width = (image_width - filter_width + 2pad) / stride + 1
        return int((input_size - filter_size + 2 * zero_padding) / stride + 1)

# filter类保存了卷积层的参数以及梯度  并且实现了用梯度下降算法来更新参数
class Filter(object):
    def __init__(self, width, height, depth):
        '''
        卷积核filter的宽 高 深度
        权重初始化为一个很小的值 偏置项初始化为0 是一个常用策略
        '''
        self.weights = np.random.uniform(-1e-4, 1e-4, (depth, height, width))
        self.bias = 0
        self.weights_gradient = np.zeros(self.weights.shape)
        self.bias_gradient = 0

    def __repr__(self):
        return 'filter weights:\n{}\nbias:\n{}'.format(repr(self.weights),
                                                       repr(self.bias))

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def update(self, learning_rate):
        '''
        每个filter的更新权重w:  w = w - (n * w_gradient)
        '''
        self.weights -= learning_rate * self.weights_gradient
        self.bias -= learning_rate * self.bias_gradient

class MaxPoolingLayer(object):
    def __init__(self, input_width, input_height, channel_number,
                 filter_width, filter_height, stride):
        self.input_width = input_width
        self.input_height = input_height
        self.channel_number = channel_number
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.stride = stride
        self.output_width = int((input_width - filter_width) / self.stride + 1)
        self.output_height = int((input_height - filter_height) / self.stride + 1)
        self.output_array = np.zeros((self.channel_number, self.output_height, self.output_width))

    def forward(self, input_array):
        '''
        前向计算  输入是特征图 根据步长S 和max pooling的filter大小x*x
        在input上 每一个x*x取出最大的神经元的值 作为一个输出值
        '''
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    self.output_array[d, i, j] = (
                        get_region(input_array[d], i, j,
                                   self.filter_width,
                                   self.filter_height,
                                   self.stride).max()
                    )

    def backward(self, input_array, sensitivity_array):
        '''
        反向计算  误差传递 从这一层传递给上一层
        对于max pooling来说 就是把本层的sensitivity_array中每个值返回个对应上一层x*x中的最大值的神经元
        而上一层其他的误差项值都是0
        '''
        self.delta_array = np.zeros(input_array.shape)
        for d in range(self.channel_number):
            for i in range(self.output_height):
                for j in range(self.output_width):
                    region_array = get_region(input_array[d], i, j,
                                              self.filter_width,
                                              self.filter_height,
                                              self.stride)
                    row, col = get_max_index(region_array)
                    self.delta_array[d,
                                     i * self.stride + row,
                                     j * self.stride + col] = sensitivity_array[d, i, j]

# ReLu激活函数
class ReLuActivator(object):
    def forward(self, weighted_input):
        return max(0, weighted_input)
        # return weighted_input   这是identity激活函数 f(x) = x

    def backward(self, output):
        return 1 if output > 0 else 0

# 卷积层的梯度检查
def init_convolution_test():
    # 输入图像 3 * 5 * 5
    input_array = np.array(
        [[[0, 1, 1, 0, 2],
          [2, 2, 2 ,2 ,1],
          [1, 0, 0, 2, 0],
          [0, 1, 1, 0, 0],
          [1, 2, 0, 0, 2]],

         [[1, 0, 2, 2, 0],
          [0, 0, 0, 2, 0],
          [1, 2, 1, 2, 1],
          [1, 0, 0, 0, 0],
          [1, 2, 1, 1, 1]],

         [[2, 1, 2, 0, 0],
          [1, 0, 0, 1, 0],
          [0, 2, 1, 0, 1],
          [0, 1, 2, 2, 2],
          [2, 1, 0, 0, 1]]]
    )
    # sensitivity array
    sensitivity_array = np.array(
        [[[0, 1, 1],
          [2, 2, 2],
          [1, 0, 0]],

         [[1, 0, 2],
          [0, 0, 0],
          [1, 2, 1]]]
    )
    # 构造卷积层
    input_width = 5
    input_height = 5
    channel_number = 3
    filter_width = 3
    filter_height = 3
    filter_number = 2
    zero_padding = 1
    stride = 2
    activator = ReLuActivator()
    learning_rate = 0.001
    convolution_layer = ConvolutionLayer(input_width, input_height, channel_number,
                                         filter_width, filter_height, filter_number,
                                         zero_padding, stride, activator, learning_rate)
    # 给卷积核初始化3 * 3 * 3 * 2
    convolution_layer.filters[0].weights = np.array(
        [[[-1, 1, 0],
          [0, 1, 0],
          [0, 1, 1]],

         [[-1, -1, 0],
          [0, 0, 0],
          [0, -1, 0]],

         [[0, 0, -1],
          [0, 1, 0],
          [1, -1, -1]]], dtype=np.float64)
    convolution_layer.filters[0].bias = 1

    convolution_layer.filters[1].weights = np.array(
        [[[1, 1, -1],
          [-1, -1, 1],
          [0, -1, 1]],

         [[0, 1, 0],
          [-1, 0, -1],
          [-1, 1, 0]],

         [[-1, 0, 0],
          [-1, 0, 1],
          [-1, 0, 0]]], dtype=np.float64)
    return input_array, sensitivity_array, convolution_layer

def gradient_check():
    '''
    梯度检查
    '''
    # 设计一个误差函数  取所有节点输出项之和
    error_function = lambda o: o.sum()

    # 计算forward值
    input_array, sensitivity_array, convolution_layer = init_convolution_test()
    convolution_layer.forward(input_array)

    # sensitivity map  是一个全1数组
    sensitivity_array = np.ones(convolution_layer.output_array.shape,
                                dtype=np.float64)
    # 计算梯度
    convolution_layer.backward(input_array, sensitivity_array, ReLuActivator())
    # 检查梯度
    epsilon = 1e-3
    # 取一个filter来遍历 深度 行 列
    for d in range(convolution_layer.filters[0].weights_gradient.shape[0]):
        for i in range(convolution_layer.filters[0].weights_gradient.shape[1]):
            for j in range(convolution_layer.filters[0].weights_gradient.shape[2]):
                convolution_layer.filters[0].weights[d, i, j] += epsilon
                convolution_layer.forward(input_array)
                error1 = error_function(convolution_layer.output_array)

                convolution_layer.filters[0].weights[d, i, j] -= 2 * epsilon
                convolution_layer.forward(input_array)
                error2 = error_function(convolution_layer.output_array)

                expect_gradient = (error1 - error2) / (2 * epsilon)
                convolution_layer.filters[0].weights[d, i, j] += epsilon
                print('weights({},{},{}): expected - actual {} -- {}'.format(
                    d, i, j, expect_gradient, convolution_layer.filters[0].weights_gradient[d, i, j]
                ))

def test_convolution():
    '''
    测试卷积层前向计算
    '''
    input_array, sensitivity_array, convolution_layer = init_convolution_test()
    convolution_layer.forward(input_array)
    print(convolution_layer.output_array)

def test_convolution_bp():
    '''
    测试卷积层反向传播
    '''
    input_array, sensitivity_array, convolution_layer = init_convolution_test()
    convolution_layer.backward(input_array, sensitivity_array, ReLuActivator())
    convolution_layer.update()
    print(convolution_layer.filters[0])
    print(convolution_layer.filters[1])

def init_pool_test():
    input_array = np.array(
        [[[1, 1, 2, 4],
          [5, 6, 7, 8],
          [3, 2, 1, 0],
          [1, 2, 3, 4]],

         [[0, 1, 2, 3],
          [4, 5, 6, 7],
          [8, 9, 0, 1],
          [3, 4, 5, 6]]], dtype=np.float64)

    sensitivity_array = np.array(
        [[[1, 2],
          [2, 4]],

         [[3, 5],
          [8, 2]]], dtype=np.float64)

    input_width = 4
    input_height = 4
    channel_number = 2
    filter_width = 2
    filter_height = 2
    stride = 2
    max_pooling_layer = MaxPoolingLayer(input_width, input_height, channel_number,
                                        filter_width, filter_height, stride)
    return input_array, sensitivity_array, max_pooling_layer

def test_max_pooling():
    '''
    测试池化层前向计算  得到最大值 降维(去掉不重要的权重)
    '''
    input_array, sensitivity_array, max_pooling_layer = init_pool_test()
    max_pooling_layer.forward(input_array)
    print('input array:\n{}\noutput array:\n{}'.format(input_array,
                                                       max_pooling_layer.output_array))

def test_max_pooling_bp():
    '''
    测试池化层反向传播
    '''
    input_array, sensitivity_array, max_pooling_layer = init_pool_test()
    max_pooling_layer.backward(input_array, sensitivity_array)
    print('input array:\n{}\nsensitivity array:\n{}\ndelta array:\n{}'.format(input_array,
                                        sensitivity_array, max_pooling_layer.delta_array))

if __name__ == '__main__':
    # gradient_check()
    # test_convolution()
    # test_convolution_bp()
    # test_max_pooling()
    test_max_pooling_bp()