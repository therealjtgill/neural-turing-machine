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
    '''
    An NTMCell that inherits from RNNCell. This inheritance was used to exploit
    the RNNCell's ability to be called by the dynamic_rnn() method, meaning
    that no custom code had to be implemented to perform dynamic unrollings of
    sequences with arbitrary lengths.
    '''

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
        Return only the size of the value that's read from the memory matrix.
        '''
        return self.M

    def __call__(self, inputs, state, scope=None):
        '''
        To support parallelization, multiple sequences from a batch are
        passed to RNNCell objects. This method takes in a tensor of shape
        [BxL1], (where B = batch size and L1 = size of an individual element of
        an input sequence) and produces an output [BxL2].
        The __call__ method is called for each element in the sequence of
        length T.

        Arguments:
          inputs - A tensor of input values [BxL1] from some piece of the
            sequence being fed into the entire recurrent unit.
          state - A tuple of tensor values consisting of the recurrent state
            from the previous timestep. The recurrent state consists of:
            1. The memory matrix
            2. The read vector from the previous timestep
            3. The write vector from the previous timestep

        Outputs:
          reads - Tensor of values that were read from the memory matrix.
          state_tuple - Tuple containing tensors relating to the recurrent
            state.
        '''

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
        '''
        Generates a state tuple containing values that are slightly biased.
        This is used to create an initial state to be fed into the RNNCell 
        before the first timestep. The read vectors are initialized with all
        memory locations being accessed uniformly. The write vectors are 
        initialized with a random element being one-hot.

        Arguments:
          batch_size - Integer size of the number of sequences in the batch.

        Outputs:
          bias_state - Tuple of numpy arrays containing the initial state for
            the RNNCell. There are N numpy arrays of size M for each memory
            location, and two numpy arrays of size N representing an initial
            value for the read and write vectors.
        '''
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
        '''
        There are several activation functions applied to the output of the 
        LSTM or FF controller, this method performs the necessary operations
        to produce the shift vector, interpolation, sharpening, key, and beta
        for the read/write operations. Also produces the add and erase vectors
        for modifying the memory matrix. This method is used outside of the
        class as well, which is why it's static.

        Arguments:
          head - Tensor of the raw output of the controller network.
          mem_size - Tuple of integers stating the size of the memory (NxM).
          shift_range - Integer that is used to determine the magnitude and
            direction of possible shifts for the read and write heads.
          axis - The axis of 'head' where splitting should occur. This is used
            for instances when 'head' is a rank 3 or rank 2 tensor. The default
            value is 1.
            (This should be eliminated to perform splitting on the last axis
            of the tensor... can probably be changed to '-1' without problems)
          style - How the head data should be reported, as a tuple or as a 
            dictionary. The tuple formulation is used for the internal 
            calculations of the NTMCell class; the dictionary form is used
            for troubleshooting.
            Possble values: "tuple" or "dict"
        '''
        N, M = mem_size
        S = shift_range
        center = int(S/2.)
        shift_bias = np.zeros(S)
        shift_bias[center+1] = 2.5
        #print(write_head_raw.get_shape(), read_head_raw.get_shape())

        # Fix the stupid head splitting; you changed it so that you wouldn't
        # have to concatenate/split crap inside of ntmagain.py

        # Number of elements in the read/write heads, respectively.
        splits = [M+S+3, 3*M+S+3]
        read_head_raw, write_head_raw = array_ops.split(head, splits,
        	axis=axis)

        write_pieces = array_ops.split(write_head_raw,
            [M, S, 1, 1, 1, M, M], axis=axis)
        read_pieces = array_ops.split(read_head_raw, [M, S, 1, 1, 1],
        	axis=axis)

        key_w, shift_w, gamma_w, beta_w, g_w, add_w, erase_w = write_pieces
            
        # Multiple operations are applied to the pieces of the write head,
        # see the original paper or this project's writeup for the breakdown.
        shift_w = nn_ops.softmax(shift_w + shift_bias)
        gamma_w = gen_math_ops.minimum(nn_ops.softplus(gamma_w) + 1, 21.)
        beta_w = nn_ops.softplus(beta_w)
        g_w = math_ops.sigmoid(g_w)
        add_w = math_ops.sigmoid(add_w)
        erase_w = math_ops.sigmoid(erase_w)

        key_r, shift_r, gamma_r, beta_r, g_r = read_pieces

        # Operations applied to the pieces of the read head.
        shift_r = nn_ops.softmax(shift_r + shift_bias)
        gamma_r = gen_math_ops.minimum(nn_ops.softplus(gamma_r) + 1, 21.)
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
    '''
    Computes the cosine similarity between tensors 'a' and 'b'. This method
    assumes that rank(a) = rank(b) = 1.

    Arguments:
      a - Rank(1) tensor.
      b - Rank(1) tensor.

    Outputs:
      cos_sim - Rank(0) tensor containing cosine similarities between tensors
        'a' and 'b'.
    '''

    dot = math_ops.reduce_sum(a*b, axis=1)

    norm_a = linalg_ops.norm(a, ord=2, axis=1)
    norm_b = linalg_ops.norm(b, ord=2, axis=1)

    # Some padding is added to the denominator to prevent 0/0 errors.
    cos_sim = math_ops.divide(dot, math_ops.add(norm_a*norm_b, 1e-8))

    return cos_sim

def circular_convolution(shift, w_i, N, S, zero_pad=False):
    '''
    This method convolves the shift output from the controller with the
    interpolated address vector, which can move the location that the address
    is pointing to.
    It's just like regular convolution, just imagine that one of the things
    that's being convolved is repeated an infinite number of times on both
    sides so that you never have to convolve with zeros. This implementation
    is kinda of tough to follow because it performs circular convolution
    between matrices (rank(2) tensors), not just vectors.

    Arguments:
      shift - Rank(2) tensor with [BxS] elements indicating the magnitude and 
        direction by which the address vector will be shifted for every batch.
        This is produced by the controller, so in most cases it won't be rows
        of one-hots.
      w_i - Rank(2) tensor with [BxN] elements corresponding to interpolated
        addresses.
      N - Integer number of memory locations in the memory matrix.
      S - Integer number of shifts that can be applied to the interpolated
        address. 
      zero_pad - Not used, left because I'm a code packrat.

    Outputs:
      op(w_conv) - An operation to stack the individual, shifted tensors into
        one big tensor.
    '''

    #shift_rev = array_ops.reverse(shift, axis=[1])
    zeros = array_ops.zeros_like(shift)
    
    split_loc = N % S
    center = int(S/2)
    #print('center:', center)
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
    '''
    Uses the various operations referenced in the paper to create a addresses
    for the read and write heads. The various steps are:
      (1) content focusing - Take softmax of cosine similarities between keys
            emitted by the controller and data stored in the memory matrix.
      (2) interpolate - Interpolate between the address at the previous
            tiemstep and the address generated from content focusing using an
            interpolation value emitted by the controller.
      (3) shift the address - Shift the mass of the address vector according to
            the shift vector emitted by the controller (circular convolution).
      (4) sharpen the address - Convolution usually results in a 'softenting'
            of previously sharp values, so the result of the convolution is
            sharpened by exponentiating all of the terms in the address and
            normalizing.

    Arguments:
      pieces - The parts of the head that are emitted in order for the network
        to generate an address (tuple containing key, shift, gamma, beta, g).
        The elements of the tuple are rank(2) tensors.
      w_prev - Rank(2) tensor of shape [BxN] containing the addresses from the
        previous timestep.
      mem_prev - Rank(3) tensor of shape [BxNxM] containing the memory matrices
        from the previous timestep. Note that the memory matrix is part of the
        NTM's recurrent state.
      N - Integer number of memory locations in the memory matrix.
      S - Integer number of shifts that can be applied to an address.
    '''


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