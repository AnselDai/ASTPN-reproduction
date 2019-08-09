import tensorflow as tf
import math

class SPP_layer:
    '''
    Spatial Pooling layer: help the network to concentrate on the region of interest
    input: the feature maps of all frames from CNN extractor, the format is:
           [batch_size, frame_size, map_height, map_width, dim(32)]
           each row is the feature map of the corresponding frame
    output: a set of image-level representation of all frames, the format is:
            [batch_size, frame_size, 1, 2720]
            where 2720 = 8*8*32 + 4*4*32 + 2*2*32 + 1*1*32
            for the feature map in 32 dims:
                we generate 4 kind of bins: 8*8, 4*4, 2*2, 1*1
                and draw them into: 1*64, 1*16, 1*4, 1*1
            finally, joint them together to get this 2720-dim vector for each frame
    '''
    def __init__(self, feature_maps):
        self.feature_maps = feature_maps
        self.res_size = [8, 4, 2, 1]
        self.batch_size = feature_maps.shape.as_list()[0]
        self.frame_size = feature_maps.shape.as_list()[1]
        self.h = feature_maps.shape.as_list()[2]
        self.w = feature_maps.shape.as_list()[3]
        self.c = feature_maps.shape.as_list()[4]
        print(feature_maps.shape.as_list())

    def build(self):
        self.feature_maps = tf.reshape(self.feature_maps, [-1, self.h, self.w, self.c])
        with tf.variable_scope('8x8_bin', reuse=tf.AUTO_REUSE) as scope:
            pool1 = self.max_pooling(self.feature_maps, math.ceil(self.h/self.res_size[0]), math.ceil(self.w/self.res_size[0]))
            bin1 = tf.reshape(pool1, [-1, self.res_size[0]*self.res_size[0]*self.c, 1])
        with tf.variable_scope('4x4_bin', reuse=tf.AUTO_REUSE) as scope:
            pool2 = self.max_pooling(self.feature_maps, math.floor(self.h/self.res_size[1]), math.floor(self.w/self.res_size[1]))
            bin2 = tf.reshape(pool2, [-1, self.res_size[1]*self.res_size[1]*self.c, 1])
        with tf.variable_scope('2x2_bin', reuse=tf.AUTO_REUSE) as scope:
            pool3 = self.max_pooling(self.feature_maps, math.ceil(self.h/self.res_size[2]), math.floor(self.w/self.res_size[2]))
            bin3 = tf.reshape(pool3, [-1, self.res_size[2]*self.res_size[2]*self.c, 1])
        with tf.variable_scope('1x1_bin', reuse=tf.AUTO_REUSE) as scope:
            pool4 = self.max_pooling(self.feature_maps, math.ceil(self.h/self.res_size[3]), math.floor(self.w/self.res_size[3]))
            bin4 = tf.reshape(pool4, [-1, self.res_size[3]*self.res_size[3]*self.c, 1])
        with tf.variable_scope('join_bin', reuse=tf.AUTO_REUSE) as scope:
            representation = tf.concat([bin1, bin2], 1)
            representation = tf.concat([representation, bin3], 1)
            representation = tf.concat([representation, bin4], 1)
            # reshape the representation into the format which RNN layer required
            representation = tf.reshape(representation, [-1, self.frame_size, 1, representation.shape.as_list()[1]])

        print('SSP layer: image_level representation')
        print(representation.shape.as_list())
        print('------------------------')

        self.image_level_representation = representation

    def max_pooling(self, x, size_h, size_w):
        return tf.nn.max_pool2d(x, ksize=[1, size_h, size_w, 1], strides=[1, size_h, size_w, 1], padding='SAME')