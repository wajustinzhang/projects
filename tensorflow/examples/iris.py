import os
import numpy as np
import tensorflow as tf
import urllib.request

IRIS_TRAINING = "iris_training.csv"
IRIS_TRAINING_URL = "http://download.tensorflow.org/data/iris_training.csv"

IRIS_TEST = "iris_test.csv"
IRIS_TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

# READ Data
if not os.path.exists(IRIS_TRAINING):
    raw = urllib.request.urlopen(IRIS_TRAINING_URL).read()
    with open(IRIS_TRAINING, 'wb') as f:
        f.write(raw)

if not os.path.exists(IRIS_TEST):
    raw = urllib.request.urlopen(IRIS_TEST_URL).read()
    with open(IRIS_TEST, 'wb') as f:
        f.write(raw)

# LOAD dataset
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TRAINING,
      target_dtype=np.int,
      features_dtype=np.float32)
test_set = tf.contrib.learn.datasets.base.load_csv_with_header(
      filename=IRIS_TEST,
      target_dtype=np.int,
      features_dtype=np.float32)

# Build classifier
feature_cols = [tf.contrib.layers.real_valued_column('', dimension=4)]
classifier = tf.contrib.learn.DNNClassifier(feature_columns = feature_cols, hidden_units=[10,20,10], n_classes=3, model_dir='./iris_model')
classifier.fit(x=training_set.data, y=training_set.target, steps=2000)

# Evaluate
train_score = classifier.evaluate(x=training_set.data, y=training_set.target, steps=1)['accuracy']
print('\n train accuracy: {0:f}\n'.format(train_score))
test_score = classifier.evaluate(x=test_set.data, y=test_set.target, steps=1)['accuracy']
print('\n test accuracy: {0:f}\n'.format(test_score))
