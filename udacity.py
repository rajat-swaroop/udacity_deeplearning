# -*- coding: utf-8 -*-
"""
Created on Sun Feb 11 22:57:32 2018

@author: 2500183
"""

"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np
import imageio
import pickle

#importing the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sklearn 
from sklearn.metrics import accuracy_score, confusion_matrix,roc_curve, auc,classification_report,precision_score,recall_score,f1_score
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import LinearSVC,SVC
from sklearn.model_selection import learning_curve,validation_curve,ShuffleSplit, cross_val_score,StratifiedKFold,GridSearchCV
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler,StandardScaler 
from sklearn.feature_selection import chi2,RFECV
from sklearn.feature_selection import SelectKBest
import time
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn import metrics 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import keras 
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.utils import np_utils
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    #pass  # TODO: Compute and return softmax(x)
    return np.exp(x)/np.sum(np.exp(x),axis=0)
    


print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()

path1 = "C:\\Users\\2500183\\Desktop\\Machine Learning\\udacity\\tensorflow\\tensorflow\\examples\\udacity\\notMNIST_large"
image_size = 28  # Pixel width and height.
pixel_depth = 255.0  # Number of levels per pixel.

image_files = [x  for x in image_files if "pickle" not in x]

def load_letter(folder, min_num_images):
  """Load the data for a single letter label."""
  image_files = os.listdir(folder)
  image_files = os.listdir(path1)
  dataset = np.ndarray(shape=(len(image_files, image_size, image_size)),dtype=np.float32)
  
  dataset = np.ndarray(shape=(len(image_files), image_size, image_size),
                         dtype=np.float32)
  
  image_file = path1+"\\A"
  
  path2 = path1+"\\A\\"+lst[0]
  
  lst = os.listdir(image_file)
  
  print(folder)
  num_images = 0
  for image in image_files:
    image_file = os.path.join(folder, image)
    try:
      image_data = (imageio.imread(path2).astype(float) - 
                    pixel_depth / 2) / pixel_depth
      if image_data.shape != (image_size, image_size):
        raise Exception('Unexpected image shape: %s' % str(image_data.shape))
      dataset[num_images, :, :] = image_data
      num_images = num_images + 1
    except (IOError, ValueError) as e:
      print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
    
  dataset = dataset[0:num_images, :, :]
  if num_images < min_num_images:
    raise Exception('Many fewer images than expected: %d < %d' %
                    (num_images, min_num_images))
    
  print('Full dataset tensor:', dataset.shape)
  print('Mean:', np.mean(dataset))
  print('Standard deviation:', np.std(dataset))
  return dataset
        
#video reading 
imageio.plugins.ffmpeg.download()


pickle.load(path1+)

path3 = path1+'\\A.pickle'
with open(path3, 'rb') as f:
        letter_set = pickle.load(f)


path3 = path1+'\\A.pickle'
with open(path3, 'rb') as f:
        letter_set = pickle.load(f)

path5 = "C:\\Users\\2500183\\Desktop\\Machine Learning\\udacity\\tensorflow\\tensorflow\\examples\\udacity\\"
path4 = path5+"notMNIST.pickle"
with open(path4,'rb') as f1:
    data = pickle.load(f1)
    
    
clf_log = LogisticRegression()
test_dataset = data['test_dataset']
test_labels = data['test_labels']
train_dataset = data['train_dataset']
train_labels = data['train_labels']
valid_dataset = data['valid_dataset']
valid_labels = data['valid_labels']


def reshape_data(dataset):
    lst = [] 
    for i in range(len(dataset)):
        y3 = dataset[i]
        #y4 = np.reshape(y3,(1,np.product(y3.shape)))
        y4 = y3.flatten()
        lst.append(y4)
    shp = [len(dataset),dataset.shape[1]*dataset.shape[2]]
    dataset2 = np.concatenate(lst).reshape(shp)
    return dataset2

train_dataset2 = reshape_data(train_dataset)

clf_log.fit(train_dataset2,train_labels)
    
    


# Use score method to get accuracy of model
score = clf_log.score(test_dataset2, test_labels)
print(score)

valid_dataset2 = reshape_data(valid_dataset)
clf_log.fit(valid_dataset2,valid_labels)

test_dataset2 = reshape_data(test_dataset)
predictions = clf_log.predict(test_dataset2)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
cm = metrics.confusion_matrix(test_labels, predictions)
print(cm)










from sklearn.datasets import load_digits
digits = load_digits()

# Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
print("Image Data Shape" , digits.data.shape)
# Print to show there are 1797 labels (integers from 0–9)
print("Label Data Shape", digits.target.shape)


import numpy as np 
import matplotlib.pyplot as plt
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
 plt.subplot(1, 5, index + 1)
 plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
 plt.title('Training: %i\n' % label, fontsize = 20)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=0)


np.reshape(a,(1,np.product(a.shape)))
y2 = test_dataset 

lst = [] 
for i in range(len(y2)):
    y3 = y2[i]
    #y4 = np.reshape(y3,(1,np.product(y3.shape)))
    y4 = y3.flatten()
    lst.append(y4)


squares = []
for x in range(10):
    squares.append(x**2)


arr = np.concatenate(lst).reshape(shp)

image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)

test_labels


test_dataset4 = test_dataset.reshape((-1, image_size * image_size)).astype(np.float32)
test_labels4 = (np.arange(num_labels) == test_labels[:,None]).astype(np.float32)

valid_dataset4 = valid_dataset.reshape((-1, image_size * image_size)).astype(np.float32)
valid_labels4 = (np.arange(num_labels) == valid_labels[:,None]).astype(np.float32)


test_labels = data['test_labels']

graph = tf.Graph()
with graph.as_default():
    tf_train_dataset = tf.constant(test_dataset4)
    tf_train_labels = tf.constant(test_labels4)
    tf_valid_dataset = tf.constant(valid_dataset4)
    tf_valid_labels = tf.constant(valid_labels4)
    #tf_test_dataset = tf.constant(test_dataset4)
    weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)


num_steps = 801

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        _, l, predictions = session.run([optimizer, loss, train_prediction])
        if (step % 100 == 0):
            print('Loss at step %d: %f' % (step, l))
            print('Training accuracy: %.1f%%' % accuracy(predictions, test_labels4))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels4))
            #print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
            
##--------------------------------------------------------------------------------
batch_size = 128

graph = tf.Graph()
with graph.as_default():
  tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset4)
  tf_test_dataset = tf.constant(test_dataset4)
  weights = tf.Variable(tf.truncated_normal([image_size * image_size, num_labels]))
  biases = tf.Variable(tf.zeros([num_labels]))
  logits = tf.matmul(tf_train_dataset, weights) + biases
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)


num_steps = 3001
with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
      offset = (step * batch_size) % (test_labels4.shape[0] - batch_size)
      batch_data = test_dataset4[offset:(offset + batch_size), :]
      batch_labels = test_labels4[offset:(offset + batch_size), :]
      feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
      _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
      if (step % 500 == 0) & (step>0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels4))
         

num_labels = 10            
batch_size = 128   
num_nodes= 1024  
L2_alpha = 0.01     
# neural net with relu 
graph = tf.Graph()
with graph.as_default():
  tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset4)
  tf_test_dataset = tf.constant(test_dataset4)
  
  weights1 = tf.Variable(tf.truncated_normal([image_size * image_size, num_nodes]))
  biases1 = tf.Variable(tf.zeros([num_nodes]))
  weights2 = tf.Variable(tf.truncated_normal([num_nodes,num_labels]))
  biases2 = tf.Variable(tf.zeros([num_labels]))
  
  logits1 = tf.matmul(tf_train_dataset, weights1) + biases1
  relu1 = tf.nn.relu(logits1)
  logits2 = tf.matmul(relu1, weights2) + biases2
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits2)
  +L2_alpha*tf.nn.l2_loss(weights1)+ L2_alpha*tf.nn.l2_loss(weights2))
  optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  train_prediction = tf.nn.softmax(logits2) 
  # Predictions for validation 
  logits_1 = tf.matmul(tf_valid_dataset, weights1) + biases1
  relu1= tf.nn.relu(logits_1)
  drop_out = tf.nn.dropout(relu1, keep_prob) 
  logits2 = tf.matmul(drop_out, weights2) + biases2
  valid_prediction = tf.nn.softmax(logits2)        
          
num_steps = 3001
with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
      offset = (step * batch_size) % (test_labels4.shape[0] - batch_size)
      batch_data = test_dataset4[offset:(offset + batch_size), :]
      batch_labels = test_labels4[offset:(offset + batch_size), :]
      feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
      _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
      if (step % 500 == 0):
          print("Minibatch loss at step %d: %f" % (step, l))
          print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
          #print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels4))
          
##---------------------------- convnets --------
num_channels = 1 #greyscale
batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

test_dataset5 = test_dataset.reshape((-1, image_size,image_size,num_channels)).astype(np.float32)
test_labels5 = (np.arange(num_labels) == test_labels[:,None]).astype(np.float32)

valid_dataset5 = valid_dataset.reshape((-1, image_size,image_size,num_channels)).astype(np.float32)
valid_labels5 = (np.arange(num_labels) == valid_labels[:,None]).astype(np.float32)


graph = tf.Graph()
with graph.as_default():
  # Input data.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset5)
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))
  layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
  layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))
  
  # Model.
  def model(data):
      conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
      hidden = tf.nn.relu(conv + layer1_biases)
      conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
      hidden = tf.nn.relu(conv + layer2_biases)
      shape = hidden.get_shape().as_list()
      reshape = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
      hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
      return tf.matmul(hidden, layer4_weights) + layer4_biases
  
  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
  
          
num_steps = 501
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = test_dataset5[offset:(offset + batch_size), :, :, :]
        batch_labels = test_labels5[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels5))
   
#--------max pooling 

graph = tf.Graph()
with graph.as_default():
  # Input data.
  tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset5)
  
  # Variables.
  layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
  layer1_biases = tf.Variable(tf.zeros([depth]))
  layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
  layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
  layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, num_hidden], stddev=0.1))
  layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
  layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
  layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))      
      
  # Model.
  def model(data):
      #conv1 = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
      conv1 = tf.nn.relu(tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME') + layer1_biases)
      pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

      
      conv2 = tf.nn.relu(tf.nn.conv2d(pool1, layer2_weights, [1, 1, 1, 1], padding='SAME') + layer2_biases)
      pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

      shape = pool2.get_shape().as_list()
      reshape = tf.reshape(pool2, [shape[0], shape[1] * shape[2] * shape[3]])
      
      fc1 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
      #fc1_drop = tf.nn.dropout(fc1, keep_prob=1)
      y_conv = tf.nn.softmax(tf.matmul(fc1, layer4_weights) + layer4_biases)
      return y_conv

  # Training computation.
  logits = model(tf_train_dataset)
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits))
    
  # Optimizer.
  optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)
  valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    
      
num_steps = 501
with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print('Initialized')
    for step in range(num_steps):
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = test_dataset5[offset:(offset + batch_size), :, :, :]
        batch_labels = test_labels5[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
        if (step % 50 == 0):
            print('Minibatch loss at step %d: %f' % (step, l))
            print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
            print('Validation accuracy: %.1f%%' % accuracy(valid_prediction.eval(), valid_labels5))    
    
    
#---nlp 
from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE   
    

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)


def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with zipfile.ZipFile(filename) as f:
    data = tf.compat.as_str(f.read(f.namelist()[0])).split()
  return data
  
words = read_data(filename)
print('Data size %d' % len(words))


vocabulary_size = 50000

def build_dataset(words):
  count = [['UNK', -1]]
  count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
  dictionary = dict()
  for word, _ in count:
    dictionary[word] = len(dictionary)
  data = list()
  unk_count = 0
  for word in words[0:20]:
    if word in dictionary:
        index = dictionary[word]
    else:
      index = 0  # dictionary['UNK']
      unk_count = unk_count + 1
    data.append(index)
  count[0][1] = unk_count
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
  return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  # Hint to reduce memory.
    

data_index = 0

def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  for i in range(batch_size // num_skips):
    target = skip_window  # target label at the center of the buffer
    targets_to_avoid = [ skip_window ]
    for j in range(num_skips):
      while target in targets_to_avoid:
        target = random.randint(0, span - 1)
      targets_to_avoid.append(target)
      batch[i * num_skips + j] = buffer[skip_window]
      labels[i * num_skips + j, 0] = buffer[target]
    buffer.append(data[data_index])
    data_index = (data_index + 1) % len(data)
  return batch, labels

data_index = 0
num_skips, skip_window = (2, 1)



print('data:', [reverse_dictionary[di] for di in data[:8]])

print('data:', [reverse_dictionary[di] for di in data[:8]])

for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])


input_list = ['all', 'this', 'happened', 'more', 'or', 'less']

def find_bigrams(input_list):
  bigram_list = []
  for i in range(len(input_list)-1):
      bigram_list.append((input_list[i], input_list[i+1]))
  return bigram_list

def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])


zip(*[input_list[i::n] for i in range(n)]).




batch_size = 128
embedding_size = 128 # Dimension of the embedding vector.
skip_window = 1 # How many words to consider left and right.
num_skips = 2 # How many times to reuse an input to generate a label.
# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. 
valid_size = 16 # Random set of words to evaluate similarity on.
valid_window = 100 # Only pick dev samples in the head of the distribution.
valid_examples = np.array(random.sample(range(valid_window), valid_size))
num_sampled = 64 # Number of negative examples to sample.

graph = tf.Graph()
with graph.as_default(), tf.device('/cpu:0'):
  # Input data.
  train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
  valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
  # Variables.
  embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
  softmax_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],stddev=1.0 / math.sqrt(embedding_size)))
  softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
  # Model.
  # Look up embeddings for inputs.
  embed = tf.nn.embedding_lookup(embeddings, train_dataset)
  # Compute the softmax loss, using a sample of the negative labels each time.
  loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                               labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))
  # Optimizer.
  # Note: The optimizer will optimize the softmax_weights AND the embeddings.
  # This is because the embeddings are defined as a variable quantity and the
  # optimizer's `minimize` method will by default modify all variable quantities 
  # that contribute to the tensor it is passed.
  # See docs on `tf.train.Optimizer.minimize()` for more details.
  optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
  
  # Compute the similarity between minibatch examples and all embeddings.
  # We use the cosine distance:
  norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
  normalized_embeddings = embeddings / norm
  valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
  similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))



num_steps = 100001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print('Initialized')
  average_loss = 0
  for step in range(num_steps):
    batch_data, batch_labels = generate_batch(
      batch_size, num_skips, skip_window)
    feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
    _, l = session.run([optimizer, loss], feed_dict=feed_dict)
    average_loss += l
    if step % 2000 == 0:
      if step > 0:
        average_loss = average_loss / 2000
      # The average loss is an estimate of the loss over the last 2000 batches.
      print('Average loss at step %d: %f' % (step, average_loss))
      average_loss = 0
    # note that this is expensive (~20% slowdown if computed every 500 steps)
    if step % 10000 == 0:
      sim = similarity.eval()
      for i in range(valid_size):
        valid_word = reverse_dictionary[valid_examples[i]]
        top_k = 8 # number of nearest neighbors
        nearest = (-sim[i, :]).argsort()[1:top_k+1]
        log = 'Nearest to %s:' % valid_word
        for k in range(top_k):
          close_word = reverse_dictionary[nearest[k]]
          log = '%s %s,' % (log, close_word)
        print(log)
  final_embeddings = normalized_embeddings.eval()


num_points = 400

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])


