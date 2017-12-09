import tensorflow as tf

class vanilla_CNN(object):
    def conv2d(self, x, W):
        """conv2d는 full stride를 가진 2d convolution layer를 반환(return)한다."""
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_4x4(self, x):
        return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')

    def max_pool_2x2(self, x):
        """특징들(feature map)을 2X만큼 downsample한다."""
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def weight_variable(self, shape):
        """주어진 shape에 대한 weight variable을 정의한다."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        """주어진 shape에 대한 bias variable을 정의한다. 초기화는 tf.global_variables_initializer()로 별도로 해주어야 한다 """
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    def window_func(self, w):
        start = w
        end = w - window_size
        X_matrix = tf.reshape(X_t[start:end], (-1, window_size))
        return nn.forward_prop(X_matrix, Theta1, bias1, Theta2, bias2)
    
    def Defining_CNN(self, x, input_size, cascade=('',)):
        """숫자를 분류하기 위한 Deep Neural Networks 그래프를 생성한다.
        인자들(Args):
        x: (N_examples, 784) 차원을 가진 input tensor
        리턴값들(Returns):
        tuple (y, keep_prob). y는 (N_examples, 10)형태의 숫자(0-9) tensor이다.
        keep_prob는 dropout을 위한 scalar placeholder이다.
        """
        # Convolutional Neural Netwokrs(CNNs)를 위한 reshape.
        # 마지막 차원(dimension)은 특징들("features")을 나타낸다.-이 코드에서는 이미지가 grayscale이라 일차원이지만, RGB 이미지라면 3차원, RGBA라면 4차원 이미지 일 것이다.
        x_image = tf.reshape(x, [-1,input_size,input_size,4,1])
        if cascade[0] == 'InputCas':
            x_image = tf.concat(0, [x_image, cascade[1]])
            W_conv_Local1 = self.weight_variable([7, 7, 9, 64])
        else:
            # Local convolutional layer - 4개의 grayscale 이미지를 64개의 특징들(feature)으로 맵핑(maping)한다.
            W_conv_Local1 = self.weight_variable([7, 7, 4, 64])        
        
        b_conv_Local1 = self.bias_variable([64])
        h_conv_Local1 = tf.contrib.layers.maxout(self.conv2d(x_image, W_conv_Local1) + b_conv_Local1)

        # Pooling layer - 4X만큼 downsample한다.
        h_pool_Local1 = self.max_pool_4x4(h_conv_Local1)
        if cascade[0] == 'LocalCas':
            h_pool_Local1 = tf.concat(0, [h_pool_Local1, cascade[1]])
            W_conv_Local2 = self.weight_variable([3, 3, 69, 64])
        else:
            W_conv_Local2 = self.weight_variable([3, 3, 64, 64])    
        
        b_conv_Local2 = self.bias_variable([64])
        h_conv_Local2 = tf.contrib.layers.maxout(self.conv2d(h_pool_Local1, W_conv_Local2) + b_conv_Local2)
        h_pool_Local2 = self.max_pool_2x2(h_conv_Local2)

        # Global convolutional layer -- 4개의 특징들(feature)을 160개의 특징들(feature)로 맵핑(maping)한다.
        W_conv_Global1 = self.weight_variable([13, 13, 4, 160])
        b_conv_Global1 = self.bias_variable([160])
        h_conv_Global1 = tf.contrib.layers.maxout(self.conv2d(x_image, W_conv_Global1) + b_conv_Global1)

        h_concat = tf.concat(0, [h_conv_Global1, h_pool_Local2])
        if cascade[0] == 'MFCas':
            h_concat = tf.concat(0, [h_concat, cascade[1]])
            W_conv_preOut = self.weight_variable([21, 21, 229, 5])
        else:
            W_conv_preOut = self.weight_variable([21, 21, 224, 5])    
        
        b_conv_preOut = self.bias_variable([5])

        y_conv = self.conv2d(h_concat, W_conv_preOut) + b_conv_preOut
    
        return y_conv

