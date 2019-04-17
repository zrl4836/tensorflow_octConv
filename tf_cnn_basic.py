import tensorflow as tf


def BN(data, bn_momentum=0.9, name=None):
    return tf.layers.batch_normalization(data, momentum=bn_momentum, name=('%s__bn' % name))


def AC(data, name=None):
    return tf.nn.relu(data, name=('%s__relu' % name))


def BN_AC(data, momentum=0.9, name=None):
    bn = BN(data=data, name=name)
    bn_ac = AC(data=bn, name=name)
    return bn_ac


def Conv(data, num_filter, kernel, stride=(1, 1), pad='valid', name=None, no_bias=False, w=None, b=None, attr=None,
         num_group=1):
    if w is None:
        conv = tf.layers.conv2d(inputs=data, filters=num_filter, kernel_size=kernel,
                                strides=stride, padding=pad, name=('%s__conv' % name), use_bias=no_bias)
    else:
        if b is None:
            conv = tf.layers.conv2d(data=data, num_filter=num_filter, kernel_size=kernel,
                                    stride=stride, padding=pad, name=('%s__conv' % name), use_bias=no_bias,
                                    kernel_initializer=w)
        else:
            conv = tf.layers.conv2d(data=data, num_filter=num_filter, kernel_size=kernel,
                                    stride=stride, padding=pad, name=('%s__conv' % name), use_bias=True,
                                    kernel_initializer=w, bias_initializer=b)
    return conv


# - - - - - - - - - - - - - - - - - - - - - - -
# Standard Common functions < CVPR >
def Conv_BN(data, num_filter, kernel, pad, stride=(1, 1), name=None, w=None, b=None, no_bias=False, attr=None,
            num_group=1):
    cov = Conv(data=data, num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name,
               w=w, b=b, no_bias=no_bias, attr=attr)
    cov_bn = BN(data=cov, name=('%s__bn' % name))
    return cov_bn


def Conv_BN_AC(data, num_filter, kernel, pad, stride=(1, 1), name=None, w=None, b=None, no_bias=False, attr=None,
               num_group=1):
    cov_bn = Conv_BN(data=data, num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride,
                     name=name, w=w, b=b, no_bias=no_bias, attr=attr)
    cov_ba = AC(data=cov_bn, name=('%s__ac' % name))
    return cov_ba


# - - - - - - - - - - - - - - - - - - - - - - -
# Standard Common functions < ECCV >
def BN_Conv(data, num_filter, kernel, pad, stride=(1, 1), name=None, w=None, b=None, no_bias=False, attr=None,
            num_group=1):
    bn = BN(data=data, name=('%s__bn' % name))
    bn_cov = Conv(data=bn, num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name,
                  w=w, b=b, no_bias=no_bias, attr=attr)
    return bn_cov

def AC_Conv(data, num_filter, kernel, pad, stride=(1, 1), name=None, w=None, b=None, no_bias=False, attr=None,
            num_group=1):
    ac = AC(data=data, name=('%s__ac' % name))
    ac_cov = Conv(data=ac, num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride, name=name,
                  w=w, b=b, no_bias=no_bias, attr=attr)
    return ac_cov


def BN_AC_Conv(data, num_filter, kernel, pad, stride=(1, 1), name=None, w=None, b=None, no_bias=False, attr=None,
               num_group=1):
    bn = BN(data=data, name=('%s__bn' % name))
    ba_cov = AC_Conv(data=bn, num_filter=num_filter, num_group=num_group, kernel=kernel, pad=pad, stride=stride,
                     name=name, w=w, b=b, no_bias=no_bias, attr=attr)
    return ba_cov

def Pooling(data, pool_type='avg', kernel=(2, 2),pad='valid', stride=(2, 2), name=None):
    if pool_type == 'avg':
        return tf.layers.average_pooling2d(inputs=data, pool_size=kernel, strides=stride, padding=pad, name=name)
    elif pool_type == 'max':
        return tf.layers.max_pooling2d(inputs=data, pool_size=kernel, strides=stride, padding=pad, name=name)

def ElementWiseSum(x, y, name=None):
    return tf.add(x=x, y=y, name=name)

def UpSampling(lf_conv, scale=2, sample_type='nearest',num_args=1, name=None):
    return tf.keras.layers.UpSampling2D(size=(scale, scale), name=name)(lf_conv)
