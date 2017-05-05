from __future__ import print_function

import collections
import math
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

class NTMCell(RNNCell):

    def __init__(self, mem_size, shift_range=3):
        self.N, self.M = mem_size
        self.shift_range = shift_range
        self._num_units = self.M*self.N + 2*self.N

    @property
    def state_size(self):
        '''
        State includes the memory matrix, and address vectors for the read
        and write heads. These values influence the matrix and addresses at
        the next time step.
        '''
        return self.N*(self.M,) + (self.N, self.N)

    @property
    def output_size(self):
        '''
        Return only the value that's read from the memory matrix.
        '''
        return self.M

    def __call__(self, inputs, state, scope=None):

        M = self.M
        N = self.N
        S = self.shift_range

        with vs.variable_scope(scope or 'ntm_cell'):
            #write_head, read_head = array_ops.split(inputs, [3*M+S+3, M+S+3],
            #    axis=1)
            mem_prev = array_ops.stack(state[0:-2], axis=1)

            w_read_prev = state[-2]
            w_write_prev = state[-1]

            write_pieces, read_pieces = self.head_pieces(inputs, (N, M), S)

            w_write = generate_address(write_pieces[0:5], w_write_prev,
                mem_prev, N, S)
            w_read = generate_address(read_pieces, w_read_prev,
                mem_prev, N, S)

            erase = array_ops.expand_dims(write_pieces[-1], axis=2)
            add = array_ops.expand_dims(write_pieces[-2], axis=2)

            w_write_ = array_ops.expand_dims(w_write, axis=2)

            erase_box = math_ops.matmul(
                w_write_, array_ops.transpose(erase, perm=[0, 2, 1]))
            add_box = math_ops.matmul(
                w_write_, array_ops.transpose(add, perm=[0, 2, 1]))

            mem_new = mem_prev*(1. - erase_box) + add_box

            read_w_ = array_ops.expand_dims(w_read, axis=1)

            reads = array_ops.squeeze(math_ops.matmul(read_w_, mem_new))
            state_tuple = tuple(array_ops.unstack(mem_new, axis=1)) + \
                (w_read, w_write)

        return reads, state_tuple

    def bias_state(self, batch_size):
        state_size = self.state_size
        start_bias = int(np.random.rand()*self.N/2.)
        
        bias_state = [
            np.abs(np.random.rand(batch_size, s))
            for s in state_size[0:-2]
        ]

        normal = np.zeros((batch_size, state_size[-1]))
        normal += 1./float(state_size[-1])
        one_hot = np.zeros((batch_size, state_size[-1]))
        one_hot[:,start_bias] = 1.
        #for i in range(batch_size):
        #   hot_index = int(np.random.rand()*self.N/2.)
        #   one_hot[i, hot_index] = 1.
        bias_state.append(normal.copy())
        bias_state.append(one_hot.copy())

        return tuple(bias_state)

    @staticmethod
    def head_pieces(head, mem_size, shift_range, axis=1, style='tuple'):
        N, M = mem_size
        S = shift_range
        center = int(S/2.)
        shift_bias = np.zeros(S)
        shift_bias[center+1] = 2.5
        #print(write_head_raw.get_shape(), read_head_raw.get_shape())

        # Fix the stupid head splitting; you changed it so that you wouldn't
        # have to concatenate/split crap inside of ntmagain.py

        splits = [M+S+3, 3*M+S+3]
        read_head_raw, write_head_raw = array_ops.split(head, splits,
        	axis=axis)

        write_pieces = array_ops.split(write_head_raw,
            [M, S, 1, 1, 1, M, M], axis=axis)
        read_pieces = array_ops.split(read_head_raw, [M, S, 1, 1, 1],
        	axis=axis)

        key_w, shift_w, gamma_w, beta_w, g_w, add_w, erase_w = write_pieces
            
        shift_w = nn_ops.softmax(shift_w + shift_bias)
        gamma_w = gen_math_ops.minimum(nn_ops.softplus(gamma_w) + 1, 21.)
        #gamma_w = 50.*math_ops.sigmoid(gamma_w/50.) + 1.
        beta_w = nn_ops.softplus(beta_w)
        g_w = math_ops.sigmoid(g_w)
        add_w = math_ops.sigmoid(add_w)
        erase_w = math_ops.sigmoid(erase_w)

        key_r, shift_r, gamma_r, beta_r, g_r = read_pieces

        shift_r = nn_ops.softmax(shift_r + shift_bias)
        gamma_r = gen_math_ops.minimum(nn_ops.softplus(gamma_r) + 1, 21.)
        #gamma_r = 50*math_ops.sigmoid(gamma_r/50.) + 1.
        beta_r = nn_ops.softplus(beta_r)
        g_r = math_ops.sigmoid(g_r)

        if style=='tuple':
            write_head = (key_w, shift_w, gamma_w, beta_w, g_w,
            	add_w, erase_w)
            read_head = (key_r, shift_r, gamma_r, beta_r, g_r)
        else:
            write_head = \
            {
                'key' : key_w,
                'shift' : shift_w,
                'gamma' : gamma_w,
                'beta' : beta_w,
                'g' : g_w,
                'add' : add_w,
                'erase' : erase_w,
            }

            read_head = \
            {
                'key' : key_r,
                'shift' : shift_r,
                'gamma' : gamma_r,
                'beta' : beta_r,
                'g' : g_r,
            }

        return write_head, read_head

def cosine_similarity(a, b):

    dot = math_ops.reduce_sum(a*b, axis=1)

    norm_a = linalg_ops.norm(a, ord=2, axis=1)
    norm_b = linalg_ops.norm(b, ord=2, axis=1)

    cos_sim = math_ops.divide(dot, math_ops.add(norm_a*norm_b, 1e-8))

    return cos_sim

def circular_convolution(shift, w_i, N, S, zero_pad=False):

    #shift_rev = array_ops.reverse(shift, axis=[1])
    zeros = array_ops.zeros_like(shift)
    
    split_loc = N % S
    center = int(S/2)
    print('center:', center)
    #center = 1

    if not zero_pad:
        num_tiles = max(int(N/S), 0)
        if num_tiles > 0:
            shift_tile = array_ops.tile(shift, [1, num_tiles])
        else:
            shift_tile = shift

        if split_loc > 0:
            tack = array_ops.split(shift, [split_loc, -1], axis=1)[0]
            shift_long = array_ops.concat([shift_tile, tack], axis=1)

    else:
        num_tiles = max(int((N - S)/S), 0)
        if num_tiles > 0:
            zeros_tile = array_ops.tile(zeros, [1, num_tiles])
        else:
            zeros_tile = zeros

        if split_loc > 0:
            tack = array_ops.split(zeros, [split_loc, -1], axis=1)[0]
            shift_long = array_ops.concat([shift, zeros_tile, tack], axis=1)

    #shift_rev_ = array_ops.reverse(shift_long, axis=[1])
    center_split = array_ops.split(shift_long, [center, -1], axis=1)
    shift_rev_ = array_ops.concat([center_split[1], center_split[0]], axis=1)
    shift_rev = array_ops.reverse(shift_rev_, axis=[1])

    circ = []
    for j in range(N):
        loc = (j + 1) % N
        shift_split = array_ops.split(shift_rev, [N-loc, loc], axis=1)
        circ.append(array_ops.concat([shift_split[1], shift_split[0]], axis=1))

    w_conv = [math_ops.reduce_sum(w_i*c, axis=1) for c in circ]

    return array_ops.stack(w_conv, axis=1)

def generate_address(pieces, w_prev, mem_prev, N, S):

    key, shift, gamma, beta, g = pieces

    w_c_arg = [cosine_similarity(m, key) \
        for m in array_ops.unstack(mem_prev, axis=1)]

    w_c_arg = array_ops.stack(w_c_arg, axis=1)

    w_c = nn_ops.softmax(beta*w_c_arg)

    w_i = g*w_c + (1. - g)*w_prev

    w_conv = circular_convolution(shift, w_i, N, S, True)

    w_sharp = math_ops.pow(w_conv, gamma)

    w = w_sharp/math_ops.reduce_sum(w_sharp, axis=1, keep_dims=True)

    return w