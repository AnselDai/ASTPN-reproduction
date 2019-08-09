import os
import numpy as np
import random
from PIL import Image

iLIDS_VID_PATH = os.getcwd() + '/Dataset/i-LIDS-VID/i-LIDS-VID/sequences/'

class DataProcessor:
    def __init__(self, data_path_1, data_path_2, labels, type, frame_size, img_h, img_w, img_c):
        self.data_path_1 = data_path_1
        self.data_path_2 = data_path_2
        self.labels = labels
        self.max_batch = data_path_1.shape[1]
        self.type = type
        self.cur_index = 0
        self.frame_size = frame_size
        self.img_h = img_h
        self.img_w = img_w
        self.img_c = img_c
        self.g_batch_start_index = 0
        self.p_batch_start_index = 0

    def next_batch(self, batch_size=100):
        if self.type == 'train':
            return self.train_next_batch(batch_size)

        elif self.type == 'test':
            print('next test batch')
            return self.test_next_batch(batch_size)

    def get_all_gallery(self, batch_size=50):
        img_h = 128
        img_w = 64
        img_c = 3
        data = np.zeros((batch_size, self.frame_size, img_h, img_w, img_c))
        labels = np.zeros(((batch_size, 1)))
        for b in range(batch_size):
            i = self.g_batch_start_index
            self.g_batch_start_index += 1
            if self.g_batch_start_index >= self.max_batch:
                self.g_batch_start_index = 0
            max_frame = len(self.data_path_1[0][i])
            frame_index = random.randint(0, max_frame - self.frame_size)
            for j in range(self.frame_size):
                path = self.data_path_1[:, i][0][frame_index + j]
                img = np.array(Image.open(path))
                data[b, j] = img
            labels[b] = self.labels[0][i]
        print('g batch index: {}'.format(self.g_batch_start_index))
        return data, labels

    def get_all_probes(self, batch_size=50):
        img_h = 128
        img_w = 64
        img_c = 3
        data = np.zeros((batch_size, self.frame_size, img_h, img_w, img_c))
        labels = np.zeros(((batch_size, 1)))
        for b in range(batch_size):
            i = self.p_batch_start_index
            self.p_batch_start_index += 1
            if self.p_batch_start_index >= self.max_batch:
                self.p_batch_start_index = 0
            max_frame = len(self.data_path_2[0][i])
            frame_index = random.randint(0, max_frame-self.frame_size)
            for j in range(self.frame_size):
                path = self.data_path_2[:, i][0][frame_index+j]
                img = np.array(Image.open(path))
                data[b, j] = img
            labels[b] = self.labels[0][i]
        print('p batch index: {}'.format(self.p_batch_start_index))
        return data, labels

    def train_next_batch(self, batch_size):
        '''
        return a batch of train data pairs
        :param batch_size: the pairs for train
        :return:
            1. [batch_size, frame_size] cam1
            2. [batch_size, frame_size] cam2
            3. [batch_size, 1] label of cam1
            4. [batch_size, 1] label of cam2
        method:
        generate batch_size pairs
        each pairs are consist of 4 parts:
        1. a sequence of frame_size of the randomly select person in camera 1
        2. a sequence of frame_size of the person in camera 2
        3. the label of person in camera1
        4. the label of person in camera2
        What's more, the pairs should be positive and negative alternately
        '''
        positive_pair = True # True means select positive pair, False means select negative pairs
        cam1_data = np.zeros((batch_size, self.frame_size, self.img_h, self.img_w, self.img_c))
        cam2_data = np.zeros((batch_size, self.frame_size, self.img_h, self.img_w, self.img_c))
        label1_data = np.zeros((batch_size, 1))
        label2_data = np.zeros((batch_size, 1))

        for batch in range(batch_size):
            # get the index, data, and label of person in cam 1
            person1_index = random.randint(0, self.max_batch - 1)
            person1_data_path_cam1 = self.data_path_1[:, person1_index][0]
            max_frame_cam1 = len(person1_data_path_cam1)
            person1_frame_start_index_cam1 = random.randint(0, max_frame_cam1-self.frame_size)
            for frame1_cam1 in range(self.frame_size):
                path = person1_data_path_cam1[person1_frame_start_index_cam1 + frame1_cam1]
                img = np.array(Image.open(path))
                cam1_data[batch, frame1_cam1] = img
            label1_data[batch] = self.labels[0][person1_index]
            # get the index, data, and label of person in cam 2
            if positive_pair:
                # select positive pair
                # choose the frames randomly of the same person person1_index in cam 2
                person1_data_path_cam2 = self.data_path_2[:, person1_index][0]
                max_frame_cam2 = len(person1_data_path_cam2)
                person1_frame_start_index_cam2 = random.randint(0, max_frame_cam2-self.frame_size)
                for frame1_cam2 in range(self.frame_size):
                    path = person1_data_path_cam2[person1_frame_start_index_cam2 + frame1_cam2]
                    img = np.array(Image.open(path))
                    cam2_data[batch, frame1_cam2] = img
                label2_data[batch] = self.labels[0][person1_index]
                positive_pair = False # change to select negative pair
            else:
                # select negative pair, the person is different from person1
                person2_index = person1_index
                while person1_index == person2_index:
                    person2_index = random.randint(0, self.max_batch-1)
                person2_data_path_cam2 = self.data_path_2[:, person2_index][0]
                max_frame_cam2 = len(person2_data_path_cam2)
                person2_frame_start_index = random.randint(0, max_frame_cam2-self.frame_size)
                for frame2_cam2 in range(self.frame_size):
                    path = person2_data_path_cam2[person2_frame_start_index + frame2_cam2]
                    img = np.array(Image.open(path))
                    cam2_data[batch, frame2_cam2] = img
                label2_data[batch] = self.labels[0][person2_index]
                positive_pair = True # change to select negative pair
        return cam1_data, cam2_data, label1_data, label2_data

    def test_next_batch(self, batch_size):
        pass

class DataReader:
    def __init__(self, frame_size=16):
        cam1, cam2, labels = self.read_data()
        num = labels.shape[1]
        self.frame_size = frame_size
        self.img_h, self.img_w, self.img_c = self.get_image_size(cam1[0][0][0])
        train_test_ratio = 0.5
        train = DataProcessor(data_path_1=cam1[:, 0:int(num*train_test_ratio)],
                              data_path_2=cam2[:, 0:int(num*train_test_ratio)],
                              labels=labels[:, 0:int(num*train_test_ratio)],
                              type='train',
                              frame_size=frame_size,
                              img_h=self.img_h,
                              img_w=self.img_w,
                              img_c=self.img_c
                              )
        test = DataProcessor(data_path_1=cam1[:, int(num*train_test_ratio):num],
                             data_path_2=cam2[:, int(num*train_test_ratio):num],
                             labels=labels[:, int(num*train_test_ratio):num],
                             type='test',
                             frame_size=frame_size,
                             img_h=self.img_h,
                             img_w=self.img_w,
                             img_c=self.img_c
                             )
        self.train = train
        self.test = test
        self.train_size = int(num*train_test_ratio)
        self.test_size = int(num*(1-train_test_ratio))

    def get_image_size(self, path):
        img = np.array(Image.open(path))
        return img.shape[0], img.shape[1], img.shape[2]

    def get_image_shape(self):
        return [self.frame_size, self.img_h, self.img_w, self.img_c]

    def generate_labels(self, cam_dir):
        person_dirs = os.listdir(iLIDS_VID_PATH + cam_dir + '/')
        person_dirs.sort()
        labels = []
        for p_dir in person_dirs:
            labels.append(int(p_dir[6:len(p_dir)]))
        return labels

    def read_data(self):
        cam1_pics_path = []
        cam2_pics_path = []
        # 读取i-LIDS-VID数据集
        camera_dirs = os.listdir(iLIDS_VID_PATH)
        if len(camera_dirs) == 0:
            print('ERROR::DATASET::NO CAMERA DIRECTORY')
            return False
        labels = self.generate_labels(camera_dirs[0])
        for label in labels:
            for cam_dir in camera_dirs:
                pics_name = os.listdir(iLIDS_VID_PATH + cam_dir + '/' + 'person' + str(label).zfill(3))
                pics_name.sort()
                if pics_name[0] == '.DS_Store':
                    pics_name = pics_name[1:len(pics_name)]
                pics_path = []
                for name in pics_name:
                    path = iLIDS_VID_PATH + cam_dir + '/person' + str(label).zfill(3) + '/' + name
                    pics_path.append(path)
                if cam_dir == 'cam1':
                    cam1_pics_path.append(pics_path)
                elif cam_dir == 'cam2':
                    cam2_pics_path.append(pics_path)
        cam1_pics_path = np.array([cam1_pics_path])
        cam2_pics_path = np.array([cam2_pics_path])
        labels = np.array([labels])
        return cam1_pics_path, cam2_pics_path, labels