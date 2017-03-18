from __future__ import print_function

import collections
import math

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables
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
	def __init__(self, mem_size, slim_output=False):
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
		#print('m n', M, N)
		with vs.variable_scope(scope or "ntm_cell"):
			write_piece, read_piece = array_ops.split(inputs, [3*M+N+3, -1], axis=1)
			mem_prev = array_ops.stack(state[0:-2], axis=1)
			#print('state len:', len(state))
			#print('mem_prev:', mem_prev)
			read_w_prev = state[-1]
			write_w_prev = state[-2]

			def cos_sim(a, b):
				dot = math_ops.reduce_sum(a*b, axis=1)
				norm = linalg_ops.norm(a, ord=2, axis=1) * \
					linalg_ops.norm(b, ord=2, axis=1)

				return math_ops.divide(dot, norm)
			
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

				circ = []
				for j in range(N):
					shift_split = array_ops.split(shift, (j,N-j), axis=1)[::-1]
					circ.append(array_ops.concat(shift_split, axis=1))
				#print('circ:', circ)

				#circ_shift = array_ops.stack(circ, axis=1)
				#w_conv = math_ops.matmul(circ_shift,
				#	array_ops.expand_dims(w_i, axis=2))

				w_conv = []
				for c in circ:
					w_conv.append(math_ops.reduce_sum(w_i*c, axis=1))

				w_conv = array_ops.stack(w_conv, axis=1)
				#print('w_conv:', w_conv)
				w_sharp = math_ops.pow(w_conv, gamma)
				#print('w_sharp:', w_sharp)

				w = w_sharp/math_ops.reduce_sum(w_sharp, axis=1,
					keep_dims=True)
				#print('w returned from inner func:', w)
				return w

			# Get the addresses from the write head.
			write_pieces = array_ops.split(write_piece,
				[M, N, 1, 1, 1, M, M], axis=1)
			write_w = generate_address(write_pieces[0:-2], write_w_prev)
			erase, add = write_pieces[-2], write_pieces[-1]

			# Get the addresses from the read head.
			read_pieces = array_ops.split(read_piece, [M, N, 1, 1, 1], axis=1)
			read_w = generate_address(read_pieces, read_w_prev)

			# Generate the new memory matrices for each batch id.
			write_w_tiled = array_ops.expand_dims(write_w, axis=2)
			write_w_tiled = array_ops.tile(write_w_tiled, [1, 1, M])

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
