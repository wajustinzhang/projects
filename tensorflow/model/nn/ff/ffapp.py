import sys
import ffnn

class ffapp:
    def __index__(self):
        pass


    if __name__ == '__main__':
        configfile = sys.argv[0]
        if configfile is None:
            configfile = 'ffconfig.json'

        model = ffnn(configfile).define_loss(train_label).start_training()
        model.evaluate(train_label)
        model.evaluate(test_label)
