'''
This script trains a lenet-like cnn using dataset i created.
It is a tiny project for practice, and does not guarantee any thing.
'''

import theano
from convolutional_mlp import LeNetConvPoolLayer
from mlp import HiddenLayer
from logistic_sgd import LogisticRegression

import theano.tensor as T
import matplotlib.pyplot as plt
import numpy
import random
import scipy
import scipy.io
import cPickle

# note: to show the image correctly, use
#   img = img.reshape([50,50], order=2) or
#   img = img.reshape([50,50], order='F')
# note: the dataset i generated using matlab is in the shape of [sample, num_sample]
#   this is not compatible with theano, which requires a [num_sample, sample input] input,
#   remember to transpose input if you use my model.

# prepare the input data and labels
# load data
data = scipy.io.loadmat('built_from_mis.mat')
train_data = data['data']
train_label = data['label']
sample_num = len(train_data[0])

indices = random.sample(xrange(sample_num), sample_num)
train_data = train_data[:, indices]
train_label = train_label[:, indices]

# devide the data set into two parts
test_data = train_data[:, 8000:-1]
test_data = test_data.transpose([1, 0])  # transpose data for the reason i declared before
test_label = train_label[:, 8000:-1]

train_data = train_data[:, indices][:, 0:8000]
train_data = train_data.transpose([1, 0])  # transpose data for the reason i declared before
train_label = train_label[:, indices][:, 0:8000]
# define variables to be used

rng = numpy.random.RandomState(3510)
# train_value = theano.shared(char_data, allow_downcast=True, dtype=theano.config.floatX, borrow=True)
# train_label = theano.shared(label_data, allow_downcast=True, dtype=theano.config.floatX, borrow=True)

input = T.matrix('input')
label = T.ivector('label')
batch_size = 100
nkerns = [10, 25]

layer0_input = input.reshape((batch_size, 1, 50, 50))

# layer 0: convolution
# input: batch_size * 1 * 50 * 50
# conv width: (50 - 5)/3 + 1 = 16
# pool width: 16 / 2 = 8
# output: batch_size * nkerns[0] * 8 * 8
layer0 = LeNetConvPoolLayer(
    rng,
    input=layer0_input,
    image_shape=(batch_size, 1, 50, 50),
    filter_shape=(nkerns[0], 1, 5, 5),
    poolsize=(2, 2),
    stride=(3, 3)
)

# layer 1: convolution
# input: batch_size * nkerns[0] * 8 * 8
# conv width: (8 - 4) + 1 = 5
# pool width: 5 / 1 = 5
# output: batch_size * nkerns[1] * 5 * 5
layer1 = LeNetConvPoolLayer(
    rng,
    input=layer0.output,
    image_shape=(batch_size, nkerns[0], 8, 8),
    filter_shape=(nkerns[1], nkerns[0], 4, 4),
    poolsize=(1, 1),
    stride=(1, 1)
)

# layer 2: FC
# input: batch_size * nkerns[1] * 5 * 5
# output: batch_size * 500
layer2_input = layer1.output.flatten(2)
layer2 = HiddenLayer(
    rng,
    inputs=layer2_input,
    n_in=nkerns[1] * 5 * 5,
    n_out=500,
    activation=T.tanh
)

# layer 3: LogisticRegression
# input: 500 * 1
# output: 36 * 1

layer3 = LogisticRegression(input=layer2.output, n_in=500, n_out=36)
cost = layer3.negative_log_likelihood(label)

# build the model
learning_rate = 0.03

params = layer3.params + layer2.params + layer1.params + layer0.params
grads = T.grad(cost, params)

updates = [
    (param_i, param_i - learning_rate * grad_i)
    for param_i, grad_i in zip(params, grads)
    ]

train_model = theano.function(
    inputs=[input, label],
    outputs=cost,
    updates=updates,
    on_unused_input='warn',
    allow_input_downcast=True,
)

test_model = theano.function(
    inputs=[input, label],
    outputs=layer3.errors(label),
    on_unused_input='warn',
    allow_input_downcast=True,
)

# start training
# i fixed epoch to 200 without validation, 200 is enough for the model to work properly
max_epoch = 200
max_batch = train_data.shape[0] / batch_size
best_cost = numpy.inf  # randomize the image and label array before each epoch

for epoch_count in range(0, max_epoch):
    print 'epoch #{}'.format(epoch_count)
    for batch_count in range(0, max_batch):
        input_data = train_data[batch_count * batch_size:(batch_count + 1) * batch_size, :]
        input_label = train_label[:, batch_count * batch_size:(batch_count + 1) * batch_size]

        this_cost = train_model(input_data, input_label[0])
        if this_cost < best_cost:
            best_cost = this_cost
    print '\t max_cost = {}, this_cost = {}'.format(best_cost, this_cost)
    learning_rate /= 1.5

print 'training completed.'

# rebuild a new net for real-world use (no batch input)
batch_size = 1
nkerns = [20, 50]

layer0_u_input = input.reshape((batch_size, 1, 50, 50))

# layer 0: convolution
# input: batch_size * 1 * 50 * 50
# conv width: (50 - 5)/3 + 1 = 16
# pool width: 16 / 2 = 8
# output: batch_size * nkerns[0] * 8 * 8
layer0_u = LeNetConvPoolLayer(
    rng,
    input=layer0_u_input,
    image_shape=(batch_size, 1, 50, 50),
    filter_shape=(nkerns[0], 1, 5, 5),
    poolsize=(2, 2),
    stride=(3, 3),
    W=layer0.W.get_value(),
    b=layer0.b.get_value(),
)

# layer 1: convolution
# input: batch_size * nkerns[0] * 8 * 8
# conv width: (8 - 4) + 1 = 5
# pool width: 5 / 1 = 5
# output: batch_size * nkerns[1] * 5 * 5
layer1_u = LeNetConvPoolLayer(
    rng,
    input=layer0_u.output,
    image_shape=(batch_size, nkerns[0], 8, 8),
    filter_shape=(nkerns[1], nkerns[0], 4, 4),
    poolsize=(1, 1),
    stride=(1, 1),
    W=layer1.W.get_value(),
    b=layer1.b.get_value(),
)

# layer 2: FC
# input: batch_size * nkerns[1] * 5 * 5
# output: batch_size * 500
layer2_u_input = layer1_u.output.flatten(2)
layer2_u = HiddenLayer(
    rng,
    inputs=layer2_u_input,
    n_in=nkerns[1] * 5 * 5,
    n_out=500,
    activation=T.tanh,
    W=layer2.W.get_value(),
    b=layer2.b.get_value(),
)

# layer 3: LogisticRegression
# input: 500 * 1
# output: 36 * 1
layer3_u = LogisticRegression(input=layer2_u.output, n_in=500, n_out=36, W=layer3.W.get_value(), b=layer3.b.get_value())

forward_only_model = theano.function(
    inputs=[input],
    outputs=layer3_u.p_y_given_x,
    on_unused_input='warn',
    allow_input_downcast=True,
)

p = forward_only_model(test_data[0:1])
pred = p[0].argmax()
print 'predict = {}, label = {}'.format(
    chr((pred - 10 + ord('A'))
        if (pred > 9)
        else (pred - 0 + ord('0'))),
    chr((test_label[0][0] - 10 + ord('A'))
        if (test_label[0][0] > 9)
        else (test_label[0][0] - 0 + ord('0'))))
print 'confidence = {}'.format(p[0].max())

# save the model to file
f = open('model.dat', 'wb')
cPickle.dump(params, f)
f.close()

# after finishing this, run test.py to have a try.
