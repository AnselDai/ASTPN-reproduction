from DataReader.Dataset import DataReader
from Layers.ASTPN_network import ASTPN_Network

DATASET = DataReader()
ASTPN = ASTPN_Network(DATASET.get_image_shape())
ASTPN.build()
ASTPN.train(DATASET)
# ASTPN.get_train_accuracy(DATASET)