import tensorflow as tf

class ATP_layer:
    '''
    Attentive Pooling Network: help network get more precise temporal information of the data
    input: the time-level matrices of 2 frame sequences, which is passed through the same RNN layer sharing the same parameter
           the format of input matrices P and G is:
           [batch_size, frame_size, hidden_num]
    process: 1. get A matrix: A = PUG^T, the U matrix is a parameter needed to learn, and U is [hidden_num, hidden_num]
                so the A is [hidden_num, hidden_num]
             2. generate tp and tg for the 2 sequences
                tp: column-wise max-pooling of A [frame_size, 1]
                tg: row-wise max-pooling of A    [frame_size, 1]
             3. get vp and vg for the 2 sequences: here is dot product
                vp = P.Softmax(tp)
                vg = G.Softmax(tg)
    output: vp and vg, both of them are [frame_size, 1] metrics
    '''
    def __init__(self, P, G):
        # [frame_size, hidden_num]
        self.P_matrix = P
        self.G_matrix = G
        self.frame_size = P.shape.as_list()[1]
        self.hidden_num = P.shape.as_list()[2]

    def build(self):
        with tf.variable_scope('ATP_U', reuse=tf.AUTO_REUSE) as scope:
            U = self.U_variable([self.hidden_num, self.hidden_num])
        with tf.variable_scope('ATP_A', reuse=tf.AUTO_REUSE) as scope:
            A = tf.nn.tanh(tf.matmul(tf.matmul(self.P_matrix, U), tf.transpose(self.G_matrix, perm=[0, 2, 1])))
            # reshape A to the format of max pooling function from [batch_size, batch_size] to [1, batch_size, batch_size, 1]
            trans_A = tf.reshape(A, [-1, self.frame_size, self.frame_size, 1])
        with tf.variable_scope('ATP_c_wise', reuse=tf.AUTO_REUSE) as scope:
            t_p = self.max_pooling(trans_A, 1, self.frame_size)
            # reshape tp back to our required format
            t_p = tf.reshape(t_p, [-1, self.frame_size, 1])
        with tf.variable_scope('ATP_r_wise', reuse=tf.AUTO_REUSE) as scope:
            t_g = self.max_pooling(trans_A, self.frame_size, 1)
            t_g = tf.reshape(t_g, [-1, self.frame_size, 1])
        with tf.variable_scope('ATP_VP', reuse=tf.AUTO_REUSE) as scope:
            v_p = tf.matmul(tf.transpose(self.P_matrix, perm=[0, 2, 1]), tf.nn.softmax(t_p))
        with tf.variable_scope('ATP_VG', reuse=tf.AUTO_REUSE) as scope:
            v_g = tf.matmul(tf.transpose(self.G_matrix, perm=[0, 2, 1]), tf.nn.softmax(t_g))
        self.vp = v_p
        self.vg = v_g

        print('ATP layer: precise temporal information')
        print(self.vp.shape.as_list())
        print('-------------------------')

    def U_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def max_pooling(self, x, size_h, size_w):
        return tf.nn.max_pool2d(x, ksize=[1, size_h, size_w, 1], strides=[1, size_h, size_w, 1], padding='SAME')