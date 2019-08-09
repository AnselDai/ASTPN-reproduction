from Layers.CNN_layer.CNN import CNN_layer
from Layers.SPP_layer.SPP import SPP_layer
from Layers.RNN_layer.RNN import RNN_layer
from Layers.ATP_layer.ATP import ATP_layer
import tensorflow as tf
from pathlib import Path
import numpy as np

class ASTPN_Network:
    def __init__(self, input_shape, batch_size=50, learning_rate=0.0001, m=20, epoch=50, hidden_num=200):
        self.input_shape = input_shape
        self.batch_size = batch_size
        # parameter
        self.learning_rate = learning_rate
        self.m = m
        self.epoch = epoch
        self.hidden_num = hidden_num
        self.model_name = "lr_" + str(learning_rate) + "_m_" + str(m) + "_epoch_" + str(epoch) + "_hidden_num_" + str(hidden_num)
        self.model_path = "./models/" + self.model_name + "/" + self.model_name + ".ckpt"
        self.model_exist = False

    def init_placeholder(self):
        with tf.variable_scope('input_x1') as scope:
            self.x1 = tf.placeholder(tf.float32, [None, self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3]])
        with tf.variable_scope('input_x2') as scope:
            self.x2 = tf.placeholder(tf.float32, [None, self.input_shape[0], self.input_shape[1], self.input_shape[2], self.input_shape[3]])
        with tf.name_scope('input_y1') as scope:
            self.y1 = tf.placeholder(tf.float32, [None, 1])
        with tf.name_scope('input_y2') as scope:
            self.y2 = tf.placeholder(tf.float32, [None, 1])

    def build(self):
        self.init_placeholder()
        with tf.variable_scope('siamese', reuse=tf.AUTO_REUSE) as scope:
            P = self.siamese_network(self.x1, self.hidden_num)
            G = self.siamese_network(self.x2, self.hidden_num)
        with tf.variable_scope('ATP', reuse=tf.AUTO_REUSE) as scope:
            ATP = ATP_layer(P, G)
            ATP.build()
            self.vp = ATP.vp
            self.vg = ATP.vg

    def train(self, dataset):
        if self.model_exist:
            return
        with tf.name_scope('train') as scope:
            loss = self.loss_func()
            optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(loss)
        with tf.Session() as sess:
            tf.initialize_all_variables().run()
            for e in range(self.epoch):
                x1_set, x2_set, y1_set, y2_set = dataset.train.next_batch(batch_size=self.batch_size)
                _loss, _ = sess.run([loss, optimizer], feed_dict={
                    self.x1: x1_set,
                    self.x2: x2_set,
                    self.y1: y1_set,
                    self.y2: y2_set
                })
                print("{}/{}: loss: {}".format(e, self.epoch, _loss))
            all_models_dir = Path("./models")
            if not all_models_dir.is_dir():
                all_models_dir.mkdir()
            model_dir = Path("./models/" + self.model_name)
            if not model_dir.is_dir():
                model_dir.mkdir()
            saver = tf.train.Saver()
            saver.save(sess, self.model_path)

    def predict(self):
        print('predict')
        distances = self.predict_distance()
        result = tf.argmin(distances)
        print(result)
        return result, distances[result[0]]

    def get_train_accuracy(self, dataset):
        '''
        calculate the accuracy of train dataset
        :return: accuracy of the predict result
        '''
        if not Path('./models/' + self.model_name).is_dir():
            self.train(dataset)
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, self.model_path)

            distances = self.predict_distance()
            # predict_a_frame = tf.argmin(distances)
            # dist = distances[predict_a_frame]

            true_num = 0
            batch_size = 50
            iteration = int(dataset.train_size / batch_size)

            for t in range(iteration):
                gallery, g_labels = dataset.train.get_all_gallery(batch_size)
                for i in range(batch_size):
                    input_x_set = np.tile(np.reshape(gallery[i], [1, gallery[i].shape[0], gallery[i].shape[1], gallery[i].shape[2], gallery[i].shape[3]]), [batch_size, 1, 1, 1, 1])

                    person = None
                    dist_value = None
                    for j in range(iteration):
                        probes, p_labels = dataset.train.get_all_probes(batch_size)
                        distance_arr = sess.run([distances], feed_dict={
                            self.x1: input_x_set,
                            self.x2: probes
                        })
                        distance_arr = np.array(distance_arr)
                        distance_arr = np.reshape(distance_arr, (distance_arr.shape[1]))
                        print(distance_arr)
                        predict_index = np.argmin(distance_arr)
                        distance = distance_arr[predict_index]
                        if dist_value == None:
                            person = p_labels[predict_index]
                            dist_value = distance
                        elif distance < dist_value:
                            dist_value = distance
                            person = p_labels[predict_index]
                        print('distance_' + str(j) + ': {}'.format(distance))
                    print('predict person: {}, expect person: {}'.format(person, g_labels[i]))
                    if g_labels[i] == person:
                        print("right")
                        true_num += 1
                    else:
                        print('wrong')
                    print('--------------------')
            accuracy = true_num / dataset.train_size
            print("The accuracy of train set is {}".format(accuracy))


    def siamese_network(self, input, hidden_num):
        CNN_extractor = CNN_layer(input=input)
        CNN_extractor.build()

        SPP = SPP_layer(CNN_extractor.feature_maps)
        SPP.build()

        RNN_extractor = RNN_layer(SPP.image_level_representation, hidden_num)
        RNN_extractor.build()

        return RNN_extractor.time_level_representation

    def loss_func(self):
        self.siamese_loss()
        return tf.reduce_mean(self.loss_vector)

    def siamese_loss(self):
        not_equal_relation = tf.abs(tf.nn.l2_normalize(tf.subtract(self.y1, self.y2), 1))
        equal_relation = tf.abs(not_equal_relation - 1)
        equal_result = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.vp, self.vg)), 1) + 1e-8)
        not_euqal_result = tf.maximum(0., self.m - equal_result)
        E = tf.add(tf.multiply(equal_relation, equal_result), tf.multiply(not_equal_relation, not_euqal_result))
        E = tf.reshape(E, [-1, 1, 1])
        I_vp = tf.nn.softmax(self.vp)
        I_vg = tf.nn.softmax(self.vg)
        _loss = E + I_vp + I_vg
        self.loss_vector = tf.reduce_mean(_loss, 1)

    def predict_distance(self):
        result = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.vp, self.vg)), 1) + 1e-8)
        I_vp = tf.nn.softmax(self.vp)
        I_vg = tf.nn.softmax(self.vg)
        print(result)
        print(I_vp)
        result = tf.reshape(result, [-1, 1, 1]) + I_vp + I_vg
        result = tf.reduce_mean(result, 1)
        return result