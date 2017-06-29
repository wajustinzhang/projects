import json
import decimal
import tensorflow as tf

class config:
    def __init__(self, fileName):
        self.fileName = fileName

    def readConfig(self):
        with open(self.fileName, 'r') as f:
            return self.__validateconfig__(json.load(f))

    def __validateconfig__(self, config):
        if config is None:
            config = {}

        # environment related configs
        data_folder = config['data_folder']
        if data_folder is None:
            config['data_folder'] = './tmp'

        log_folder = config['log_folder']
        if log_folder is None:
            config['log_folder'] = './log'

        if tf.gfile.Exists(log_folder):
            tf.gfile.DeleteRecursively(log_folder)
            tf.gfile.MakeDirs(log_folder)

        # hyperparameters
        rate = config['learning_rate']
        if rate is None or rate > 1 or rate < 0:
            print ('Learning rate should not be a valid value between 0 ~ 1')
            config['learning_rate'] = 0.01
        else:
            config['learning_rate'] = json.loads(config['learning_rate'], parse_float=decimal.Decimal)

        # Iterating steps
        steps = config['steps']
        if steps is None:
            config['steps'] = 2000  #set default
        else:
            config["steps"] = int(config["steps"])

        batch_size = config["mini_batch_size"]
        if batch_size is None:
            config["mini_batch_size"] = 100
        else:
            config["mini_batch_size"] = int(config["mini_batch_size"])

        # neural network hidden layers definition, passing as a tuple like (128, 32) means a nn with 2 hidden layers,
        # first hidden layer has 128 neurons, second hidden layer has 32 neurons
        layers = config["layers"]
        if layers is None:
            config["layers"] = (128, 32)
        else:
            config["layers"] = tuple(config["layers"])

        return config
