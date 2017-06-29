import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.examples.tutorials.mnist import mnist

class mnist:
    def getDataSet(self, inputDataDir):
        self.dataset =  input_data.read_data_sets(inputDataDir, None)

    def fill_feed_dict(self, images_pl, labels_pl):
        images_feed, labels_feed = self.dataset.next_batch(FLAGS.batch_size,
                                               FLAGS.fake_data)