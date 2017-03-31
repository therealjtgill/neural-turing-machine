from __future__ import print_function

import collections
import math
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell

class NTMCell(RNNCell):
	'''
	Neural Turing Machine Cell.

	Both the read and write heads on the controller produce address vectors
	that are dependent on the memory matrix, so the outputs of both heads
	are processed.

	No trainable variables are introduced here, only manipulations of the
	outputs of the read/write heads and the memory matrix occur.

	This class is meant to be wrapped in a 'dynamic_rnn' method.
	'''
	def __init__(self, mem_size, shift_range=2, slim_output=False):
		'''
		Args:
			mem_size: Tuple containing dimensions of the memory matrix, in 
			  the paper expressed as (N,M). This value determines the sizes
			  of all other vectors.
		  --*Future*--
		    slim_output: Boolean value controlling what data is reported in
		      the output of the NTM. 'False' allows the memory matrix, 
		      read/write addresses, and values read from the matrix to all be
		      reported in the cell's output. 'True' limits the output to 
		      containing only values read from the matrix
		'''
		if len(mem_size) != 2:
			raise ValueError('Incorrect number of memory indices. Received ' + \
				str(len(mem_size)) + 'but expected 2.')
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
		For now, return all final values that are calculated.
		'''
		#return self.M*self.N + 2*self.N
		return self.N*(self.M,) + (self.N, self.N, self.M)

	def __call__(self, inputs, state, scope=None):
		'''
		Args:
			inputs: 2D tensor of size [batch_size, input_size] where 
			  'input_size' is the number of bits composing an instruction.
			state: Tuple of (N + 2) 2D tensors
			  [batch_size, M] --> N of these; rows of the unstacked memory
			  [batch_size, N] --> 2 of these; write address and read address
			scope: Name to give the subgraph used to contain these ops.

		Returns:
			(tuple of two tuples)
			tuple_0: A combination of all items in the NTM's recurrent state
			  and the values read from memory.
			  (unstacked memory, read addresses, write addresses, read values)
			tuple_1: All items in the NTM's recurrent state.
			  (unstacked memory, read addresses, write addresses)

			Note: Not sure if the NTM's output is allowed to be a tuple.
		'''
		M = self.M
		N = self.N
		S = self.shift_range
		#print('m n', M, N)
		with vs.variable_scope(scope or "ntm_cell"):
			write_head, read_head = array_ops.split(inputs, [3*M+S+3, M+S+3],
				axis=1)
			mem_prev = array_ops.stack(state[0:-2], axis=1)
			#print('state len:', len(state))
			#print('mem_prev:', mem_prev)
			read_w_prev = state[-1]
			write_w_prev = state[-2]

			def cos_sim(a, b):
				'''
				Compute the cosine similarity between vectors 'a' and 'b'.

				Args:
					a, b: tensors of size [batch_size, M] whose cosine
					  similarity will be computed.

				Returns:
					(a o b)/(|a||b|)
				'''
				dot = math_ops.reduce_sum(a*b, axis=1)
				#norm = linalg_ops.norm(a, ord=2, axis=1) * \
				#	linalg_ops.norm(b, ord=2, axis=1)
				#norm_a = math_ops.sqrt(math_ops.reduce_sum(a*a, axis=1))
				#norm_b = math_ops.sqrt(math_ops.reduce_sum(b*b, axis=1))
				norm_a = linalg_ops.norm(a, ord=2, axis=1)
				norm_b = linalg_ops.norm(b, ord=2, axis=1)
				#logging_ops.Print(norm_a, [norm_a])
				#logging_ops.Print(norm_b, [norm_b])

				return math_ops.divide(dot, math_ops.add(norm_a*norm_b, 1e-3))

			def circular_convolve(shift, w_i):
				'''
				Calculates circular convolution of the shift weights and the
				interpolated address vector.
				Convert the shift array to a reversed length N array by 
				tiling up to N % S - 1 copies, and tacking on a split portion
				of the reversed array.
				EX:
				  shift = [1, 2, 3]
				  N = 11 (number of memory cells), S = 3 (length of shift)
				  shift_rev = [3, 2, 1]
				  shift_long = [3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 2  ]
				               | orig  | tile 1 | tile 2 | split |  
				'''   
				shift_rev = array_ops.reverse(shift, axis=[1])
				num_tiles = max(int(N / S), 0)
				#num_tiles = 0 if num_tiles < 0 else num_tiles
				split_loc = (N % S)


				#print('num_tiles, split_loc:', num_tiles, split_loc)

				if num_tiles > 0:
					shift_long = array_ops.tile(shift_rev, [1, num_tiles])
				else:
					shift_long = shift_rev

				if split_loc > 0:
					tack = array_ops.split(shift_rev, (split_loc, -1), axis=1)[0]
					shift_long = array_ops.concat([shift_long, tack], axis=1)

				#print('num_tiles, split_loc:', num_tiles, split_loc)
				#print('shape of shift_long:', shift_long.get_shape())
						
				circ = []
				for j in range(N):
					shift_split = array_ops.split(shift_long, (j,N-j), axis=1)[::-1]
					circ.append(array_ops.concat(shift_split, axis=1))

				#return circ
				w_conv = []
				for c in circ:
					w_conv.append(math_ops.reduce_sum(w_i*c, axis=1))

				return array_ops.stack(w_conv, axis=1)

			
			def generate_address(pieces, w_prev):
				'''
				Use the pieces provided to generate a memory address at each
				batch id.
				'''
				key, shift, gamma, beta, g = pieces

				w_c_arg = []
				for m in array_ops.unstack(mem_prev, axis=1):
					w_c_arg.append(cos_sim(m, key))

				w_c_arg = array_ops.stack(w_c_arg, axis=1)
				#print('w_c_arg:', w_c_arg)
				
				#print('w_prev', w_prev)
				#beta_exp = array_ops.expand_dims(beta, axis=2)
				w_c = nn_ops.softmax(beta*w_c_arg)

				w_i = g*w_c + (1.-g)*w_prev

				w_conv = circular_convolve(shift, w_i)
				#print('w_conv:', w_conv)
				w_sharp = math_ops.pow(w_conv, gamma)
				#print('w_sharp:', w_sharp)

				w = w_sharp / math_ops.reduce_sum(w_sharp, axis=1,
					keep_dims=True)
				#print('w returned from inner func:', w)
				return w

			# Get the addresses from the write head.
			#write_pieces = array_ops.split(write_head,
			#	[M, S, 1, 1, 1, M, M], axis=1)
			write_pieces, read_pieces = self.head_pieces(write_head,
				read_head, (N, M), S)
			write_w = generate_address(write_pieces[0:5], write_w_prev)
			read_w = generate_address(read_pieces, read_w_prev)

			#erase = math_ops.sigmoid(write_pieces[-1])
			#add = math_ops.sigmoid(write_pieces[-2])
			erase = write_pieces[-1]
			add =  write_pieces[-2]

			# Get the addresses from the read head.
			#read_pieces = array_ops.split(read_head, [M, S, 1, 1, 1], axis=1)
			#read_w = generate_address(read_pieces, read_w_prev)

			# Generate the new memory matrices for each batch id.
			write_w_exp = array_ops.expand_dims(write_w, axis=2)
			write_w_tiled = array_ops.tile(write_w_exp, [1, 1, M])

			erase_diag = array_ops.matrix_diag(erase)
			add_diag = array_ops.matrix_diag(add)

			erase_product = 1. - math_ops.matmul(write_w_tiled, erase_diag)
			add_product = math_ops.matmul(write_w_tiled, add_diag)

			mem_new = mem_prev*erase_product + add_product
			#print('memory:', mem_new)

			# Get the values read from the memory matrix.
			read_w_exp = array_ops.expand_dims(read_w, axis=1)

			#print('read/write addresses', read_w, write_w)
			reads = array_ops.squeeze(math_ops.matmul(read_w_exp, mem_new))
			state_tuple = tuple(array_ops.unstack(mem_new, axis=1)) + \
				(write_w, read_w)
			#print('returned stuff', state_tuple + (reads,))
			#print('reads:', reads)
			return state_tuple + (reads,), state_tuple

	def small_state(self, batch_size):
		state_size = self.state_size

		small_state = [
			array_ops.ones(shape=[batch_size, s])/100 \
			for s in state_size
		]

		return tuple(small_state)

	def random_state(self, batch_size):
		state_size = self.state_size
		random_state = [
			random_ops.random_normal(shape=[batch_size, s], stddev=0.01,
				mean=0.05)
			for s in state_size
		]

		return tuple(random_state)

	def bias_state(self, batch_size):
		state_size = self.state_size
		start_bias = int(np.random.rand()*self.N/2)
		
		bias_state = [
			np.abs(np.random.rand(batch_size, s)/100)
			for s in state_size[0:-2]
		]

		one_hot = np.zeros((batch_size, state_size[-1]))
		bias_state.append(one_hot)
		bias_state.append(one_hot)

		return tuple(bias_state)

	@staticmethod
	def head_pieces(write_head_raw, read_head_raw, shape, shift_range, axis=1, style='tuple'):
		N, M = shape
		S = shift_range
		print(write_head_raw.get_shape(), read_head_raw.get_shape())
		write_pieces = array_ops.split(write_head_raw,
			[M, S, 1, 1, 1, M, M], axis=axis)
		read_pieces = array_ops.split(read_head_raw, [M, S, 1, 1, 1], axis=axis)

		key_w, shift_w, gamma_w, beta_w, g_w, add_w, erase_w = write_pieces
			
		shift_w = nn_ops.softmax(shift_w)
		#gamma_w = nn_ops.softplus(gamma_w) + 1.
		gamma_w = 50.*math_ops.sigmoid(gamma_w) + 1.
		beta_w = nn_ops.softplus(beta_w)
		g_w = math_ops.sigmoid(g_w)
		add_w = math_ops.sigmoid(add_w)
		erase_w = math_ops.sigmoid(erase_w)

		key_r, shift_r, gamma_r, beta_r, g_r = read_pieces

		shift_r = nn_ops.softmax(shift_r)
		#gamma_r = nn_ops.softplus(gamma_r) + 1.
		gamma_r = 50*math_ops.sigmoid(gamma_r) + 1.
		beta_r = nn_ops.softplus(beta_r)
		g_r = math_ops.sigmoid(g_r)

		if style=='tuple':
			write_head = (key_w, shift_w, gamma_w, beta_w, g_w, add_w, erase_w)
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