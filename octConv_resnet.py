import numpy as np
import tensorflow as tf
from tf_octConv import *
from tf_cnn_basic import *
from oct_Resnet_unit import *


G = 1
alpha = 0.25
use_fp16 = True
k_sec = {2: 3, 3: 4, 4: 6, 5: 3}


def get_before_pool():

    data = tf.Variable(name="data")
    data = tf.cast(x=data, dtype=np.float16) if use_fp16 else data

    # conv1
    conv1 = Conv_BN_AC(data=data, num_filter=64, kernel=(7, 7), name='conv1', pad='same', stride=(2, 2))
    pool1 = Pooling(data=conv1, pool_type="max", kernel=(3, 3), pad='same', stride=(2, 2), name="pool1")

    # conv2
    num_in = 32
    num_mid = 64
    num_out = 256
    i = 1
    hf_conv1_x, lf_conv1_x = Residual_Unit_first(
        data=pool1,
        alpha=alpha,
        num_in=(num_in if i == 1 else num_out),
        num_mid=num_mid,
        num_out=num_out,
        name=('conv2_B%02d' % i),
        first_block=(i == 1),
        stride=((1, 1) if (i == 1) else (1, 1)))

    for i in range(2, k_sec[2] + 1):
        hf_conv2_x, lf_conv2_x = Residual_Unit(
            hf_data=(hf_conv1_x if i == 1 else hf_conv2_x),
            lf_data=(lf_conv1_x if i == 1 else lf_conv2_x),
            alpha=alpha,
            num_in=(num_in if i == 1 else num_out),
            num_mid=num_mid,
            num_out=num_out,
            name=('conv2_B%02d' % i),
            first_block=(i == 1),
            stride=((1, 1) if (i == 1) else (1, 1)))

    # conv3
    num_in = num_out
    num_mid = int(num_mid * 2)
    num_out = int(num_out * 2)
    for i in range(1, k_sec[3] + 1):
        hf_conv3_x, lf_conv3_x = Residual_Unit(
            hf_data=(hf_conv2_x if i == 1 else hf_conv3_x),
            lf_data=(lf_conv2_x if i == 1 else lf_conv3_x),
            alpha=alpha,
            num_in=(num_in if i == 1 else num_out),
            num_mid=num_mid,
            num_out=num_out,
            name=('conv3_B%02d' % i),
            first_block=(i == 1),
            stride=((2, 2) if (i == 1) else (1, 1)))


    # conv4
    num_in = num_out
    num_mid = int(num_mid * 2)
    num_out = int(num_out * 2)
    for i in range(1, k_sec[4] + 1):
        hf_conv4_x, lf_conv4_x = Residual_Unit(
            hf_data=(hf_conv3_x if i == 1 else hf_conv4_x),
            lf_data=(lf_conv3_x if i == 1 else lf_conv4_x),
            alpha=alpha,
            num_in=(num_in if i == 1 else num_out),
            num_mid=num_mid,
            num_out=num_out,
            name=('conv4_B%02d' % i),
            first_block=(i == 1),
            stride=((2, 2) if (i == 1) else (1, 1)))


    # conv5
    num_in = num_out
    num_mid = int(num_mid * 2)
    num_out = int(num_out * 2)
    i = 1
    conv5_x = Residual_Unit_last(
        hf_data=hf_conv4_x,
        lf_data=lf_conv4_x,
        alpha=alpha,
        num_in=(num_in if i == 1 else num_out),
        num_mid=num_mid,
        num_out=num_out,
        name=('conv5_B%02d' % i),
        first_block=(i == 1),
        stride=((2, 2) if (i == 1) else (1, 1)))

    for i in range(2, k_sec[5] + 1):
        conv5_x = Residual_Unit_norm(data=conv5_x,
                                     num_in=num_out,
                                     num_mid=num_mid,
                                     num_out=num_out,
                                     name=('conv5_B%02d' % i),
                                     first_block=(i == 1),
                                     stride=((2, 2) if (i == 1) else (1, 1)))

    output = tf.cast(x=conv5_x, dtype=np.float32) if use_fp16 else conv5_x
    # output
    return output


def get_linear(num_classes=10):
    before_pool = get_before_pool()
    pool5 = Pooling(data=before_pool, pool_type="avg", kernel=(7, 7), stride=(1, 1), name="global-pool")
    flat5 = tf.layers.Flatten(input=pool5, name='flatten')
    fc6 = tf.layers.dense(inputs=flat5, units=num_classes, name='classifier')
    return fc6


def get_symbol(num_classes=10):
    fc6 = get_linear(num_classes)
    softmax = tf.nn.softmax(logits=fc6, name='softmax')
    sys_out = softmax
    return sys_out
