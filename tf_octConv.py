import tensorflow as tf
from tf_cnn_basic import *

def firstOctConv(data, settings, ch_in, ch_out, name, kernel=(1,1), pad='valid', stride=(1,1)):
    alpha_in, alpha_out = settings
    hf_ch_in = int(ch_in * (1 - alpha_in))
    hf_ch_out = int(ch_out * (1 - alpha_out))

    lf_ch_in = ch_in - hf_ch_in
    lf_ch_out = ch_out - hf_ch_out

    hf_data = data

    if stride == (2, 2):
        hf_data = Pooling(data=hf_data, pool_type='avg', kernel=(2,2), stride=(2,2), name=('%s_hf_down' % name))
    hf_conv = Conv(data=hf_data, num_filter=hf_ch_out, kernel=kernel, pad=pad, stride=(1,1), name=('%s_hf_conv' % name))
    hf_pool = Pooling(data=hf_data, pool_type='avg', kernel=(2,2), stride=(2,2), name=('%s_hf_pool' % name))
    hf_pool_conv = Conv(data=hf_pool, num_filter=lf_ch_out, kernel=kernel, pad=pad, stride=(1,1), name=('%s_hf_pool_conv' % name))

    out_h = hf_conv
    out_l = hf_pool_conv
    return out_h, out_l

def lastOctConv(hf_data, lf_data, settings, ch_in, ch_out, name, kernel=(1,1), pad='valid', stride=(1,1)):
    alpha_in, alpha_out = settings
    hf_ch_in = int(ch_in * (1 - alpha_in))
    hf_ch_out = int(ch_out * (1 - alpha_out))

    if stride == (2, 2):
        hf_data = Pooling(data=hf_data, pool_type='avg', kernel=(2,2), stride=(2,2), name=('%s_hf_down' % name))
    hf_conv = Conv(data=hf_data, num_filter=hf_ch_out, kernel=kernel, pad=pad, stride=(1,1), name=('%s_hf_conv' % name))

    lf_conv = Conv(data=lf_data, num_filter=hf_ch_out, kernel=kernel, pad=pad, stride=(1,1), name=('%s_lf_conv' % name))
    out_h = hf_conv + lf_conv

    return out_h

def OctConv(hf_data, lf_data, settings, ch_in, ch_out, name, kernel=(1,1), pad='valid', stride=(1,1)):
    alpha_in, alpha_out = settings
    hf_ch_in = int(ch_in * (1 - alpha_in))
    hf_ch_out = int(ch_out * (1 - alpha_out))

    lf_ch_in = ch_in - hf_ch_in
    lf_ch_out = ch_out - hf_ch_out

    if stride == (2, 2):
        hf_data = Pooling(data=hf_data, pool_type='avg', kernel=(2,2), stride=(2,2), name=('%s_hf_down' % name))
    hf_conv = Conv(data=hf_data, num_filter=hf_ch_out, kernel=kernel, pad=pad, stride=(1,1), name=('%s_hf_conv' % name))
    hf_pool = Pooling(data=hf_data, pool_type='avg', kernel=(2,2), stride=(2,2), name=('%s_hf_pool' % name))
    hf_pool_conv = Conv(data=hf_pool, num_filter=lf_ch_out, kernel=kernel, pad=pad, stride=(1,1), name=('%s_hf_pool_conv' % name))

    lf_conv = Conv(data=lf_data, num_filter=hf_ch_out, kernel=kernel, pad=pad, stride=(1,1), name=('%s_lf_conv' % name))
    if stride == (2, 2):
        lf_upsample = lf_conv
        lf_down = Pooling(data=lf_data, pool_type='avg', kernel=(2,2), stride=(2,2), name=('%s_lf_down' % name))
    else:
        lf_upsample = UpSampling(lf_conv, scale=2, sample_type='nearest',num_args=1, name='%s_lf_upsample' % name)
        lf_down = lf_data
    lf_down_conv = Conv(data=lf_down, num_filter=lf_ch_out, kernel=kernel, pad=pad, stride=(1,1), name=('%s_lf_down_conv' % name))

    out_h = hf_conv + lf_upsample
    out_l = hf_pool_conv + lf_down_conv

    return out_h, out_l


def firstOctConv_BN_AC(data, alpha, num_filter_in, num_filter_out,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    hf_data, lf_data = firstOctConv(data=data, settings=(0, alpha), ch_in=num_filter_in, ch_out=num_filter_out, name=name, kernel=kernel, pad=pad, stride=stride)
    out_hf = BN_AC(data=hf_data, name=('%s_hf') % name)
    out_lf = BN_AC(data=lf_data, name=('%s_lf') % name)
    return out_hf, out_lf

def lastOctConv_BN_AC(hf_data, lf_data, alpha, num_filter_in, num_filter_out,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    conv = lastOctConv(hf_data=hf_data, lf_data=lf_data, settings=(alpha, 0), ch_in=num_filter_in, ch_out=num_filter_out, name=name, kernel=kernel, pad=pad, stride=stride)
    out = BN_AC(data=conv, name=name)
    return out

def octConv_BN_AC(hf_data, lf_data, alpha, num_filter_in, num_filter_out,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    hf_data, lf_data = OctConv(hf_data=hf_data, lf_data=lf_data, settings=(alpha, alpha), ch_in=num_filter_in, ch_out=num_filter_out, name=name, kernel=kernel, pad=pad, stride=stride)
    out_hf = BN_AC(data=hf_data, name=('%s_hf') % name)
    out_lf = BN_AC(data=lf_data, name=('%s_lf') % name)
    return out_hf, out_lf


def firstOctConv_BN(data, alpha, num_filter_in, num_filter_out,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    hf_data, lf_data = firstOctConv(data=data, settings=(0, alpha), ch_in=num_filter_in, ch_out=num_filter_out, name=name, kernel=kernel, pad=pad, stride=stride)
    out_hf = BN(data=hf_data, name=('%s_hf') % name)
    out_lf = BN(data=lf_data, name=('%s_lf') % name)
    return out_hf, out_lf

def lastOctConv_BN(hf_data, lf_data, alpha, num_filter_in, num_filter_out,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    conv = lastOctConv(hf_data=hf_data, lf_data=lf_data, settings=(alpha, 0), ch_in=num_filter_in, ch_out=num_filter_out, name=name, kernel=kernel, pad=pad, stride=stride)
    out = BN(data=conv, name=name)
    return out

def octConv_BN(hf_data, lf_data, alpha, num_filter_in, num_filter_out,  kernel, pad, stride=(1,1), name=None, w=None, b=None, no_bias=True, attr=None, num_group=1):
    hf_data, lf_data = OctConv(hf_data=hf_data, lf_data=lf_data, settings=(alpha, alpha), ch_in=num_filter_in, ch_out=num_filter_out, name=name, kernel=kernel, pad=pad, stride=stride)
    out_hf = BN(data=hf_data, name=('%s_hf') % name)
    out_lf = BN(data=lf_data, name=('%s_lf') % name)
    return out_hf, out_lf





