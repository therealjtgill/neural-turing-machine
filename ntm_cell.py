# Author: Jonathan Gill
# Github: /therealjtgill
'''
Implementation of a Neural Turing Machine according to the paper by Graves,
et al, from 2014.
'''

from __future__ import print_function

import numpy as np
from numpy.random import rand

from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell


# TODO(@therealjtgill) All single-character variable names need to be changed
# to something more descriptive. Add a note in the code stating how variable
# names map to the variables used in the paper.
class NTMCell(RNNCell):
  '''
  An NTMCell that inherits from RNNCell. This inheritance was used to exploit
  the RNNCell's ability to be called by the dynamic_rnn() method, meaning
  that no custom code had to be implemented to perform dynamic unrollings of
  sequences with arbitrary lengths.
  '''

  def __init__(self, mem_size, num_shifts=3):
    self.num_slots, self.num_bits = mem_size
    self.num_shifts = num_shifts
    self._num_units = self.num_bits*self.num_slots + 2*self.num_slots

  @property
  def state_size(self):
    '''
    State includes the memory matrix, and address vectors for the read
    and write heads. These values influence the matrix and addresses at
    the next time step.
    '''
    return self.num_slots*(self.num_bits,) + (self.num_slots, self.num_slots)

  @property
  def output_size(self):
    '''
    Return only the size of the value that's read from the memory matrix.
    '''
    return self.num_bits

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

    Returns:
      reads - Tensor of values that were read from the memory matrix.
      state_tuple - Tuple containing tensors relating to the recurrent
        state.
    '''

    num_bits = self.num_bits
    num_slots = self.num_slots
    num_shifts = self.num_shifts

    with vs.variable_scope(scope or 'ntm_cell'):

      mem_prev = array_ops.stack(state[0:-2], axis=1)

      w_read_prev = state[-2]
      w_write_prev = state[-1]
      mem_size = (num_slots, num_bits)

      write_pieces, read_pieces = self.head_pieces(inputs,
                                                   mem_size, num_shifts)

      w_write = generate_address(write_pieces[0:5], w_write_prev,
                                 mem_prev, num_slots, num_shifts)
      w_read = generate_address(read_pieces, w_read_prev,
                                mem_prev, num_slots, num_shifts)

      erase = write_pieces[-1]
      add = write_pieces[-2]
      mem_new = write_memory(w_write, mem_prev, erase, add)

      reads = read_memory(w_read, mem_new)
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
      batch_size - Integer size; the number of sequences in a batch.

    Returns:
      bias_state - Tuple of numpy arrays containing the initial state for
        the RNNCell. There are N numpy arrays of size M for each memory
        location, and two numpy arrays of size N representing an initial
        value for the read and write vectors.
    '''
    state_size = self.state_size
    start_bias = int(rand()*self.num_slots/2.)

    bias_state = [np.abs(rand(batch_size, s)) for s in state_size[0:-2]]

    uniform = np.zeros((batch_size, state_size[-1]))
    uniform += 1./float(state_size[-1])
    one_hot = np.zeros((batch_size, state_size[-1]))
    one_hot[:, start_bias] = 1.
    #bias_state.append(uniform.copy())
    bias_state.append(one_hot.copy())
    bias_state.append(one_hot.copy())

    return tuple(bias_state)

  @staticmethod
  def head_pieces(head, mem_size, num_shifts=3, axis=1):
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
      num_shifts - Integer that is used to determine the magnitude and
        direction of possible shifts for the read and write heads.
      axis - The axis of 'head' where splitting should occur. This is used
        for instances when 'head' is a rank 3 or rank 2 tensor. The default
        value is 1.
        (This should be eliminated to perform splitting on the last axis
        of the tensor... can probably be changed to '-1' without problems)
    '''
    num_slots, num_bits = mem_size
    _ = num_slots
    #center = int(num_shifts/2.)
    shift_bias = np.zeros(num_shifts)
    #shift_bias[center] = 2.5 # Temporarily commented out for regression
                              # testing with NP implementation.
    #print(write_head_raw.get_shape(), read_head_raw.get_shape())

    # Number of elements in the read/write heads, respectively.
    splits = [num_bits+num_shifts+3, 3*num_bits+num_shifts+3]
    read_head_raw, write_head_raw = array_ops.split(head, splits,
                                                    axis=axis)

    write_splits = [num_bits, num_shifts, 1, 1, 1, num_bits, num_bits]
    read_splits = [num_bits, num_shifts, 1, 1, 1]
    write_pieces = array_ops.split(write_head_raw, write_splits, axis=axis)
    read_pieces = array_ops.split(read_head_raw, read_splits, axis=axis)

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

    write_head = (key_w, shift_w, gamma_w, beta_w, g_w, add_w, erase_w)
    read_head = (key_r, shift_r, gamma_r, beta_r, g_r)

    return write_head, read_head

def head_pieces_tuple_to_dict(write_head, read_head):
  '''
  Converts a tuple of head pieces into a dictionary of head pieces for ease of
  use.
  '''

  write_head_dict = \
  {
      'key': write_head[0],
      'shift': write_head[1],
      'gamma': write_head[2],
      'beta': write_head[3],
      'g': write_head[4],
      'add': write_head[5],
      'erase': write_head[6],
  }

  read_head_dict = \
  {
      'key': read_head[0],
      'shift': read_head[0],
      'gamma': read_head[0],
      'beta': read_head[0],
      'g': read_head[0],
  }

  return write_head_dict, read_head_dict

def cosine_similarity(vec_a, vec_b):
  '''
  Computes the cosine similarity between tensors 'vec_a' and 'vec_b'. This
  method assumes that rank(vec_a) = rank(vec_b) = 1.

  Arguments:
    vec_a - Rank(1) tensor.
    vec_b - Rank(1) tensor.

  Returns:
    cos_sim - Rank(0) tensor containing cosine similarities between tensors
      'vec_a' and 'vec_b'.
  '''

  dot = math_ops.reduce_sum(vec_a*vec_b, axis=1)

  norm_a = linalg_ops.norm(vec_a, ord=2, axis=1)
  norm_b = linalg_ops.norm(vec_b, ord=2, axis=1)

  # Some padding is added to the denominator to prevent 0/0 errors.
  cos_sim = math_ops.divide(dot, math_ops.add(norm_a*norm_b, 1e-8))

  return cos_sim

def shift_address(shift, w_i, num_slots, num_shifts):
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
    num_slots - Integer number of memory locations in the memory matrix.
    num_shifts - Integer number of shifts that can be applied to the
      interpolated address.

  Returns:
    op(w_conv) - An operation to stack the individual, shifted tensors onto
      each other to make one big tensor.
  '''

  center = int(num_shifts/2)
  #num_tiles = max(int((N - num_shifts)/num_shifts), 0)
  #print('S:', num_shifts, 'N:', num_slots)
  #print('Center:', center)

  if num_slots > num_shifts:
    zero_pad = array_ops.split(array_ops.zeros_like(w_i),
                               [num_slots - num_shifts, -1], axis=1)[0]
    shift_long = array_ops.concat([shift, zero_pad], axis=1)
  else:
    shift_long = shift

  #print('Shift:', shift)
  #print('Shift long:', shift_long)

  #shift_rev_ = array_ops.reverse(shift_long, axis=[1])
  center_split = array_ops.split(shift_long, [center, -1], axis=1)
  shift_rev_ = array_ops.concat([center_split[1], center_split[0]], axis=1)
  shift_rev = array_ops.reverse(shift_rev_, axis=[1])
  #print('Number of memory cells:', num_slots)

  circ = []
  for j in range(num_slots):
    loc = (j + 1) % num_slots
    #print(shift_rev.shape, num_slots - loc, loc)
    shift_split = array_ops.split(shift_rev, [num_slots - loc, loc], axis=1)
    circ.append(array_ops.concat([shift_split[1], shift_split[0]], axis=1))

  w_conv = [math_ops.reduce_sum(w_i*c, axis=1) for c in circ]

  return array_ops.stack(w_conv, axis=1)

def generate_address(pieces, w_prev, mem_prev, num_slots, num_shifts):
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
    (4) sharpen the address - Convolution usually results in a 'softening'
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
    num_slots - Integer number of memory locations in the memory matrix.
    num_shifts - Integer number of shifts that can be applied to an address.

  Returns:
    w - A batch of memory addresses of shape [BxN], where each address is
      calculated according to the rules in the original NTM paper by Graves
      et al.
  '''


  key, shift, gamma, beta, interp = pieces

  w_c_arg = [cosine_similarity(m, key) \
    for m in array_ops.unstack(mem_prev, axis=1)]

  w_c_arg = array_ops.stack(w_c_arg, axis=1)

  w_c = nn_ops.softmax(beta*w_c_arg)

  w_i = interp*w_c + (1. - interp)*w_prev

  w_conv = shift_address(shift, w_i, num_slots, num_shifts)

  w_sharp = math_ops.pow(w_conv, gamma)

  address = w_sharp/math_ops.reduce_sum(w_sharp, axis=1, keep_dims=True)

  return address

def write_memory(w_write, mem_prev, erase, add):
  '''
  Uses the write address, memory matrix from the previous timestep, and the
  erase and add vectors from the controller's write head to write data to
  the memory matrix.

  Arguments:
    w_write - Rank(2) tensor of shape [BxN] whose elements at a particular
      value of B all sum to one. The write address.
    mem_prev - Rank(3) tensor of shape [BxNxM]. A batch of memory matrices.
    erase - Rank(2) tensor of shape [BxM] whose elements at a particular
      value of B are used to remove values from the memory matrix.
    add - Rank(2) tensor of shape [BxM] whose elements at a particular value
      of B are used to add values to the memory matrix.

  Returns:
    mem_new - Rank(3) matrix of shape [BxNxM], which is the old memory
      matrix with any modifications from the erase and add vectors.
  '''

  erase = array_ops.expand_dims(erase, axis=2)
  add = array_ops.expand_dims(add, axis=2)

  w_write_ = array_ops.expand_dims(w_write, axis=2)

  erase_box = \
    math_ops.matmul(w_write_,
                    array_ops.transpose(erase, perm=[0, 2, 1]))

  add_box = \
    math_ops.matmul(w_write_,
                    array_ops.transpose(add, perm=[0, 2, 1]))

  mem_new = mem_prev*(1. - erase_box) + add_box

  return mem_new

def read_memory(w_read, mem_new):
  '''
  Reads values from a batch of memory matrices according to a batch of
  read locations.

  Arguments:
    w_read - Rank(2) tensor of shape [BxN] whose elements at a particular
      value of B all sum to one. The read address.
    mem_new - Rank(3) tensor of shape [BxNxM]. A batch of memory matrices.

  Returns:
    reads - Rank(2) tensor of shape [BxM] corresponding to values that have
      been read from the batch of memory matrices.
  '''

  w_read = array_ops.expand_dims(w_read, axis=1)

  reads = array_ops.squeeze(math_ops.matmul(w_read, mem_new))
  return reads

def address_regression(pieces, w_prev, mem_prev, num_slots, num_shifts):
  '''
  Generates an address, but returns all of the intermediate steps in addition
  to the address. This is for regression tests.
  '''

  key, shift, gamma, beta, g = pieces

  w_c_arg = [cosine_similarity(m, key) \
    for m in array_ops.unstack(mem_prev, axis=1)]

  w_c_arg = array_ops.stack(w_c_arg, axis=1)

  w_c = nn_ops.softmax(beta*w_c_arg)

  w_i = g*w_c + (1. - g)*w_prev

  w_conv = shift_address(shift, w_i, num_slots, num_shifts)

  w_sharp = math_ops.pow(w_conv, gamma)

  w = w_sharp/math_ops.reduce_sum(w_sharp, axis=1, keep_dims=True)

  return [w_c, w_i, w_conv, w]
