import tensorflow as tf
from tf_cnn_basic import *
from tf_octConv import *


def Residual_Unit_norm(data, num_in, num_mid, num_out, name, first_block=False, stride=(1, 1), g=1):
    conv_m1 = Conv_BN_AC(data=data, num_filter=num_mid, kernel=(1, 1), pad='valid', name=('%s_conv-m1' % name))
    conv_m2 = Conv_BN_AC(data=conv_m1, num_filter=num_mid, kernel=(3, 3), pad='same', name=('%s_conv-m2' % name),
                         stride=stride, num_group=g)
    conv_m3 = Conv_BN(data=conv_m2, num_filter=num_out, kernel=( 1, 1), pad='valid', name=('%s_conv-m3' % name))

    if first_block:
        data = Conv_BN(data=data, num_filter=num_out, kernel=( 1, 1), pad='valid', name=('%s_conv-w1' % name),
                       stride=stride)

    outputs = ElementWiseSum(data, conv_m3, name=('%s_sum' % name))
    return AC(outputs)


def Residual_Unit_last(hf_data, lf_data, alpha, num_in, num_mid, num_out, name, first_block=False, stride=(1, 1), g=1):
    hf_data_m, lf_data_m = octConv_BN_AC(hf_data=hf_data, lf_data=lf_data, alpha=alpha, num_filter_in=num_in,
                                         num_filter_out=num_mid, kernel=( 1, 1), pad='valid',
                                         name=('%s_conv-m1' % name))
    conv_m2 = lastOctConv_BN_AC(hf_data=hf_data_m, lf_data=lf_data_m, alpha=alpha, num_filter_in=num_mid,
                                num_filter_out=num_mid, name=('%s_conv-m2' % name), kernel=(3, 3), pad='same',
                                stride=stride)
    conv_m3 = Conv_BN(data=conv_m2, num_filter=num_out, kernel=( 1, 1), pad='valid', name=('%s_conv-m3' % name))

    if first_block:
        data = lastOctConv_BN(hf_data=hf_data, lf_data=lf_data, alpha=alpha, num_filter_in=num_in,
                              num_filter_out=num_out, name=('%s_conv-w1' % name), kernel=(1, 1), pad='valid',
                              stride=stride)

    outputs = ElementWiseSum(data, conv_m3, name=('%s_sum' % name))
    outputs = AC(outputs, name=('%s_act' % name))
    return outputs


def Residual_Unit_first(data, alpha, num_in, num_mid, num_out, name, first_block=False, stride=(1, 1), g=1):
    hf_data_m, lf_data_m = firstOctConv_BN_AC(data=data, alpha=alpha, num_filter_in=num_in, num_filter_out=num_mid,
                                              kernel=( 1, 1), pad='valid', name=('%s_conv-m1' % name))
    hf_data_m, lf_data_m = octConv_BN_AC(hf_data=hf_data_m, lf_data=lf_data_m, alpha=alpha, num_filter_in=num_mid,
                                         num_filter_out=num_mid, kernel=( 3, 3), pad='same',
                                         name=('%s_conv-m2' % name), stride=stride, num_group=g)
    hf_data_m, lf_data_m = octConv_BN(hf_data=hf_data_m, lf_data=lf_data_m, alpha=alpha, num_filter_in=num_mid,
                                      num_filter_out=num_out, kernel=( 1, 1), pad='valid', name=('%s_conv-m3' % name))

    if first_block:
        hf_data, lf_data = firstOctConv_BN(data=data, alpha=alpha, num_filter_in=num_in, num_filter_out=num_out,
                                           kernel=( 1, 1), pad='valid', name=('%s_conv-w1' % name), stride=stride)

    hf_outputs = ElementWiseSum(hf_data, hf_data_m, name=('%s_hf_sum' % name))
    lf_outputs = ElementWiseSum(lf_data, lf_data_m, name=('%s_lf_sum' % name))

    hf_outputs = AC(hf_outputs, name=('%s_hf_act' % name))
    lf_outputs = AC(lf_outputs, name=('%s_lf_act' % name))
    return hf_outputs, lf_outputs


def Residual_Unit(hf_data, lf_data, alpha, num_in, num_mid, num_out, name, first_block=False, stride=(1, 1), g=1):
    hf_data_m, lf_data_m = octConv_BN_AC(hf_data=hf_data, lf_data=lf_data, alpha=alpha, num_filter_in=num_in,
                                         num_filter_out=num_mid, kernel=( 1, 1), pad='valid',
                                         name=('%s_conv-m1' % name))
    hf_data_m, lf_data_m = octConv_BN_AC(hf_data=hf_data_m, lf_data=lf_data_m, alpha=alpha, num_filter_in=num_mid,
                                         num_filter_out=num_mid, kernel=( 3, 3), pad='same',
                                         name=('%s_conv-m2' % name), stride=stride, num_group=g)
    hf_data_m, lf_data_m = octConv_BN(hf_data=hf_data_m, lf_data=lf_data_m, alpha=alpha, num_filter_in=num_mid,
                                      num_filter_out=num_out, kernel=( 1, 1), pad='valid', name=('%s_conv-m3' % name))

    if first_block:
        hf_data, lf_data = octConv_BN(hf_data=hf_data, lf_data=lf_data, alpha=alpha, num_filter_in=num_in,
                                      num_filter_out=num_out, kernel=( 1, 1), pad='valid', name=('%s_conv-w1' % name),
                                      stride=stride)

    hf_outputs = ElementWiseSum(hf_data, hf_data_m, name=('%s_hf_sum' % name))
    lf_outputs = ElementWiseSum(lf_data, lf_data_m, name=('%s_lf_sum' % name))

    hf_outputs = AC(hf_outputs, name=('%s_hf_act' % name))
    lf_outputs = AC(lf_outputs, name=('%s_lf_act' % name))
    return hf_outputs, lf_outputs
