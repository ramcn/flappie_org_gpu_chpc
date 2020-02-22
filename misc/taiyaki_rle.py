#!/usr/bin/env python3

#  Copyright 2018 Oxford Nanopore Technologies, Ltd

#  This Source Code Form is subject to the terms of the Oxford Nanopore
#  Technologies, Ltd. Public License, v. 1.0. If a copy of the License
#  was not  distributed with this file, You can obtain one at
#  http://nanoporetech.com

import argparse
import math
import re
import sys

import taiyaki
from taiyaki import helpers
from taiyaki.cmdargs import AutoBool, FileExists

parser = argparse.ArgumentParser()
parser.add_argument('--id', default='' , help='Identifier for model names')
parser.add_argument('--scale', default=False, action=AutoBool,
                    help='Correct scaling when network trained without MAD factor')
parser.add_argument('model', action=FileExists, help='Pickle to read model from')


trim_trailing_zeros = re.compile('0+p')

def small_hex(f):
    hf = float(f).hex()
    return trim_trailing_zeros.sub('p', hf)


def process_column(v, pad):
    """ process and pad """
    return [small_hex(f) for f in v] + [small_hex(0.0)] * pad


def cformatM(fh, name, X, nr=None, nc=None):
    nrq = int(math.ceil(X.shape[1] / 4.0))
    pad = nrq * 4 - X.shape[1]
    lines = map(lambda v: ', '.join(process_column(v, pad)), X)

    if nr is None:
        nr = X.shape[1]
    else:
        nrq = int(math.ceil(nr / 4.0))
    if nc is None:
        nc = X.shape[0]

    fh.write('float {}[] = {}\n'.format('__' + name, '{'))
    fh.write('\t' + ',\n\t'.join(lines))
    fh.write('};\n')
    fh.write('_Mat {} = {}\n\t.nr = {},\n\t.nrq = {},\n\t.nc = {},\n\t.stride = {},\n\t.data.f = {}\n{};\n'.format('_' + name, '{', nr, nrq, nc, nrq * 4, '__' + name, '}'))
    fh.write('const flappie_matrix {} = &{};\n\n'.format(name, '_' + name))


def cformatV(fh, name, X):
    nrq = int(math.ceil(X.shape[0] / 4.0))
    pad = nrq * 4 - X.shape[0]
    lines = ', '.join(list(map(lambda f: small_hex(f), X)) + [small_hex(0.0)] * pad)
    fh.write('float {}[] = {}\n'.format( '__' + name, '{'))
    fh.write('\t' + lines)
    fh.write('};\n')
    fh.write('_Mat {} = {}\n\t.nr = {},\n\t.nrq = {},\n\t.nc = {},\n\t.stride = {},\n\t.data.f = {}\n{};\n'.format('_' + name, '{', X.shape[0], nrq, 1, nrq * 4, '__' + name, '}'))
    fh.write('const flappie_matrix {} = &{};\n\n'.format(name, '_' + name))


def print_lstm(gru, name):
    iW = gru.lstm.weight_ih_l0
    sW = gru.lstm.weight_hh_l0
    b = gru.lstm.bias_ih_l0
    cformatM(sys.stdout, '{}iW'.format(name), iW)
    cformatM(sys.stdout, '{}sW'.format(name), sW)
    cformatV(sys.stdout, '{}b'.format(name), b.reshape(-1))


if __name__ == '__main__':
    args = parser.parse_args()
    modelid = args.id + '_'

    network = helpers.load_model(args.model)
    #  Remove first layer (DeltaSample)
    if type(network.sublayers[0]) == taiyaki.layers.DeltaSample:
        sys.stderr.write('First layer is DeltaSample, removing\n')
        network.sublayers = network.sublayers[1:]

    sys.stdout.write("""#pragma once
    #ifndef RLE_{}MODEL_H
    #define RLE_{}MODEL_H
    #include "../util.h"
    """.format(modelid.upper(), modelid.upper()))

    """ Convolution layer
    """

    filterW =  network.sublayers[0].conv.weight
    if args.scale:
        #  Scaling factor for MAD
        filterW *= 1.4826
    nfilter, _ , winlen = filterW.shape
    cformatM(sys.stdout, 'conv_rnnrf_rle_{}W'.format(modelid), filterW.reshape(-1, 1), nr = winlen * 4 - 3, nc=nfilter)
    cformatV(sys.stdout, 'conv_rnnrf_rle_{}b'.format(modelid), network.sublayers[0].conv.bias.reshape(-1))
    sys.stdout.write("#define conv_rnnrf_rle_{}stride  {}\n".format(modelid, network.sublayers[0].stride))
    sys.stdout.write("""#define {}nfilter  {}
    #define _conv_rnnrf_rle_{}winlen  {}
    """.format(modelid, nfilter, modelid, winlen))

    """  Backward GRU (first layer)
    """
    gru1 = network.sublayers[1].layer
    print_lstm(gru1, 'gruB1_rnnrf_rle_{}'.format(modelid))

    """  Forward GRU (second layer)
    """
    gru2 = network.sublayers[2]
    print_lstm(gru2, 'gruF2_rnnrf_rle_{}'.format(modelid))

    """ backward GRU(third layer)
    """
    gru3 = network.sublayers[3].layer
    print_lstm(gru3, 'gruB3_rnnrf_rle_{}'.format(modelid))

    """  Forward GRU (fourth layer)
    """
    gru4 = network.sublayers[4]
    print_lstm(gru4, 'gruF4_rnnrf_rle_{}'.format(modelid))

    """ backward GRU(fifth layer)
    """
    gru5 = network.sublayers[5].layer
    print_lstm(gru5, 'gruB5_rnnrf_rle_{}'.format(modelid))
    """ Global norm layer
    """
    gnlayer = network.sublayers[6]
    nstate = gnlayer.linear.weight.shape[0]
    cformatM(sys.stdout, 'FF_rnnrf_rle_{}W'.format(modelid), gnlayer.linear.weight)
    cformatV(sys.stdout, 'FF_rnnrf_rle_{}b'.format(modelid), gnlayer.linear.bias)

    sys.stdout.write('#endif /* RLE_{}MODEL_H */'.format(modelid.upper()))
