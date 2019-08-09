import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

class RNN_layer:
    '''
    RNN extractor: extract the temporal information from the image-level representation of a sequence of frames
    input: the image-level representation of a sequence of frames, the format is:
           [batch_size, frame_size, 1, representation_length(2720)]
    output: the raw time-level representation matrix, the format is:
            [batch_size, frame_size, hidden_num]
            where hidden_num means the number of neural cell in a RNN cell
    '''
    def __init__(self, image_level_representation, hidden_num):
        self.image_level_representation = image_level_representation
        self.hidden_num = hidden_num
        self.frame_size = image_level_representation.shape.as_list()[1]
        self.sequence_length = image_level_representation.shape.as_list()[2]
        self.representation_length = image_level_representation.shape.as_list()[3]

    def build(self):
        # 把输入转换成dynamic_rnn接受的形状：[batch_size, sequence_length, frame_size]
        '''
        Initialize a basic RNN cell with hidden_num neural cells
        Formula of RNN:
        output = new_state = f(U*input + W*s^t-1)
        default activation：f = tanh
        '''
        self.image_level_representation = tf.reshape(self.image_level_representation, [-1, self.sequence_length, self.representation_length])
        rnn_cell = tf.nn.rnn_cell.BasicRNNCell(self.hidden_num)
        output, states = tf.nn.dynamic_rnn(rnn_cell, self.image_level_representation, dtype=tf.float32)
        print(output)
        self.time_level_representation = tf.reshape(output, [-1, self.frame_size, self.hidden_num])

        print('RNN layer: temporal information')
        print(self.time_level_representation.shape.as_list())
        print('--------------------')
