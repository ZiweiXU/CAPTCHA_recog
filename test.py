'''
You can run this directly if model.dat is presented.

This script downloads a CAPTCHA from http://mis.teach.ustc.edu.cn/randomImage.do?xx and give its prediction
based on the lenet i trained.

You can modify main.py to train your own net on your own dataset.

This tiny project is based on theano's official guide.
'''


import theano
import theano.tensor as T
import scipy.misc
import PIL
import matplotlib.pyplot as plt
import Image
import httplib
import io
import numpy as np
import cPickle
from logistic_sgd import LogisticRegression
from mlp import HiddenLayer
from convolutional_mlp import LeNetConvPoolLayer


def getImage():
    url = "mis.teach.ustc.edu.cn"
    conn = httplib.HTTPConnection(url)
    conn.request("GET", "/randomImage.do?date=%271457152804424%27")
    r = conn.getresponse()
    img_data = r.read()
    conn.close()
    return img_data


# download a CAPTCHA from http://mis.teach.ustc.edu.cn
print 'grabbing image ...'
raw_img_data = getImage()
print 'fetched image ...'
stream = io.BytesIO(raw_img_data)
img = Image.open(stream)
img = scipy.misc.imresize(img, [50, 200])
imgarray = np.array(img)

# convert the RGB image to BW image
imgbw = np.array((np.mean(imgarray, axis=2) < 128).astype(int))

imgs = np.zeros([50, 50, 4]);
for i in range(0, 4):
    imgs[:, :, i] = imgbw[:, i * 50:(i + 1) * 50]

# load the trained model
f = open('model.dat', 'rb')
params = cPickle.load(f);
f.close()

input = T.matrix('input')
label = T.ivector('label')
nkerns = [20, 50]
rng = np.random.RandomState(3510)
batch_size = 1

layer0_input = input.reshape((batch_size, 1, 50, 50))
layer0 = LeNetConvPoolLayer(
    rng,
    input=layer0_input,
    image_shape=(batch_size, 1, 50, 50),
    filter_shape=(nkerns[0], 1, 5, 5),
    poolsize=(2, 2),
    stride=(3, 3),
    W=params[6].get_value(),
    b=params[7].get_value(),
)

layer1 = LeNetConvPoolLayer(
    rng,
    input=layer0.output,
    image_shape=(batch_size, nkerns[0], 8, 8),
    filter_shape=(nkerns[1], nkerns[0], 4, 4),
    poolsize=(1, 1),
    stride=(1, 1),
    W=params[4].get_value(),
    b=params[5].get_value(),
)

layer2_input = layer1.output.flatten(2)
layer2 = HiddenLayer(
    rng,
    inputs=layer2_input,
    n_in=nkerns[1] * 5 * 5,
    n_out=500,
    activation=T.tanh,
    W=params[2].get_value(),
    b=params[3].get_value(),
)

layer3 = LogisticRegression(
    input=layer2.output,
    n_in=500, n_out=36,
    W=params[0].get_value(),
    b=params[1].get_value())

forward = theano.function(
    inputs=[input],
    outputs=layer3.p_y_given_x,
    on_unused_input='warn',
    allow_input_downcast=True,
)

for i in range(0, 4):
    p = forward(imgs[:, :, i].reshape([1, 2500], order=2))
    pred = p.argmax()
    confidence = p.max()
    print 'predict = {}, confidence = {}'.format(
        chr((pred - 10 + ord('A'))
            if (pred > 9)
            else (pred - 0 + ord('0'))),
        confidence)

plt.show(plt.imshow(img))
