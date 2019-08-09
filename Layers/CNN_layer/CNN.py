import tensorflow as tf

class CNN_layer:
    '''
    A CNN extractor: to extract spatial information in a frame-sequence
    input: a sequences of frames, the format is:
          [batch_size, frame_size, img_height, img_width, img_channel]
    output: a batch of feature maps of frames, the format is:
            [batch_size, frame_size, map_height, map_width, final_dim]
            in our network, final map dimension is set to 32
    '''
    def __init__(self, input):
        '''
        Initialize the CNN network with the input sequence
        '''
        self.input = input
        self.input_shape = input.shape.as_list()

    def build(self):
        '''
        Build the CNN network
        :return: feature_maps: [batch_size, h, w, 32]
        '''
        self.input = tf.reshape(self.input, [-1, self.input_shape[2], self.input_shape[3], self.input_shape[4]])
        with tf.variable_scope('conv1', reuse=tf.AUTO_REUSE) as scope:
            W_conv1 = self.weight_variables([5, 5, 3, 16])
            b_conv1 = self.bias_variable([16])
            h_conv1 = tf.nn.leaky_relu(self.conv2d(self.input, W_conv1, 1, 'SAME') + b_conv1)
            h_pool1 = self.max_pooling(h_conv1, 2, 2)
        with tf.variable_scope('conv2', reuse=tf.AUTO_REUSE) as scope:
            W_conv2 = self.weight_variables([5, 5, 16, 32])
            b_conv2 = self.bias_variable([32])
            h_conv2 = tf.nn.leaky_relu(self.conv2d(h_pool1, W_conv2, 1, 'SAME') + b_conv2)
            h_pool2 = self.max_pooling(h_conv2, 2, 2)
        with tf.variable_scope('conv3', reuse=tf.AUTO_REUSE) as scope:
            W_conv3 = self.weight_variables([5, 5, 32, 32])
            b_conv3 = self.bias_variable([32])
            h_conv3 = tf.nn.leaky_relu(self.conv2d(h_pool2, W_conv3, 1, 'SAME') + b_conv3)
        self.feature_maps = tf.reshape(h_conv3, [-1, self.input_shape[1], h_conv3.shape.as_list()[1], h_conv3.shape.as_list()[2], h_conv3.shape.as_list()[3]])
        print('CNN layer: feature maps')
        print(self.feature_maps.shape.as_list())
        print('-----------------------')

    # weights
    def weight_variables(self, shape):
        initial = tf.random.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    # bias
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    # conv_layer
    def conv2d(self, x, W, stride, padding):
        return tf.nn.conv2d(x, W, strides=[ 1, stride, stride, 1], padding=padding)

    # maxpooling_layer
    def max_pooling(self, x, size_h, size_w):
        return tf.nn.max_pool2d(x, ksize=[1, size_h, size_w, 1], strides=[1, size_h, size_w, 1], padding='SAME')