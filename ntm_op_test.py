import tensorflow as tf
import numpy as np

stufflist = {}

#print('m n', M, N)
def head_ops(inputs, state):
	global stufflist
	write_piece, read_piece = tf.split(inputs, [3*M+S+3, -1], axis=1)
	mem_prev = tf.stack(state[0:-2], axis=1)
	#print('state len:', len(state))
	#print('mem_prev:', mem_prev)
	read_w_prev = state[-1]
	write_w_prev = state[-2]

	def cos_sim(a, b):

		dot = tf.reduce_sum(a*b, axis=1)

		norm_a = tf.norm(a, ord=2, axis=1)
		norm_b = tf.norm(b, ord=2, axis=1)

		stufflist['div'] = tf.divide(dot, tf.add(norm_a*norm_b, 1e-3))

		return tf.divide(dot, tf.add(norm_a*norm_b, 1e-3))

	def circular_convolve(shift, w_i):

		shift_rev = tf.reverse(shift, axis=[1])
		num_tiles = max(int(N / S), 0)
		#num_tiles = 0 if num_tiles < 0 else num_tiles
		split_loc = (N % S)
		if num_tiles > 0 and split_loc == 0:
			num_tiles += 1

		print('num_tiles, split_loc:', num_tiles, split_loc)

		if num_tiles > 0:
			shift_long = tf.tile(shift_rev, [1, num_tiles])
		else:
			shift_long = shift_rev

		if split_loc > 0:
			tack = tf.split(shift_rev, (split_loc, -1), axis=1)[0]
			shift_long = tf.concat([shift_long, tack], axis=1)

		print('num_tiles, split_loc:', num_tiles, split_loc)
		print('shape of shift_long:', shift_long.get_shape())
				
		circ = []
		for j in range(N):
			shift_split = tf.split(shift_long, (j,N-j), axis=1)[::-1]
			circ.append(tf.concat(shift_split, axis=1))

		#return circ
		w_conv = []
		for c in circ:
			w_conv.append(tf.reduce_sum(w_i*c, axis=1))

		return tf.stack(w_conv, axis=1)


	def generate_address(pieces, w_prev):
		'''
		Use the pieces provided to generate a memory address at each
		batch id.
		'''
		key, shift, gamma, beta, g = pieces

		w_c_arg = []
		for m in tf.unstack(mem_prev, axis=1):
			w_c_arg.append(cos_sim(m, key))

		w_c_arg = tf.stack(w_c_arg, axis=1)
		#print('w_c_arg:', w_c_arg)
		
		#print('w_prev', w_prev)
		#beta_exp = tf.expand_dims(beta, axis=2)
		w_c = tf.nn.softmax(beta*w_c_arg)
		stufflist['wc'] = w_c

		stufflist['wprev'] = tf.constant(w_prev)
		stufflist['interm'] = (1.-g)*w_prev
		w_i = g*w_c + (1.-g)*w_prev
		stufflist['wi'] = w_i

		w_conv = circular_convolve(shift, w_i)
		stufflist['wconv'] = w_conv
		#print('w_conv:', w_conv)
		w_sharp = tf.pow(w_conv, gamma)
		stufflist['wsharp'] = w_sharp
		#print('w_sharp:', w_sharp)

		w = w_sharp / tf.reduce_sum(w_sharp, axis=1,
			keep_dims=True)
		#print('w returned from inner func:', w)
		return w


	# Get the addresses from the write head.
	write_pieces = tf.split(write_piece,	[M, S, 1, 1, 1, M, M], axis=1)
	write_w = generate_address(write_pieces[0:-2], write_w_prev)
	erase, add = write_pieces[-2], write_pieces[-1]

	# Get the addresses from the read head.
	read_pieces = tf.split(read_piece, [M, S, 1, 1, 1], axis=1)
	read_w = generate_address(read_pieces, read_w_prev)

	# Generate the new memory matrices for each batch id.
	write_w_tiled = tf.expand_dims(write_w, axis=2)
	write_w_tiled = tf.tile(write_w_tiled, [1, 1, M])

	erase_diag = tf.matrix_diag(erase)
	add_diag = tf.matrix_diag(add)

	erase_product = 1. - tf.matmul(write_w_tiled, erase_diag)
	add_product = tf.matmul(write_w_tiled, add_diag)

	mem_new = mem_prev*erase_product + add_product
	#print('memory:', mem_new)

	# Get the values read from the memory matrix.
	read_w_exp = tf.expand_dims(read_w, axis=1)

	#print('read/write addresses', read_w, write_w)
	reads = tf.sigmoid(tf.squeeze(tf.matmul(read_w_exp, mem_new)))
	state_tuple = tuple(tf.unstack(mem_new, axis=1)) + (write_w, read_w)

	return reads

def bias_state(batch_size):
	
	start_bias = int(np.random.rand()*N/2)
	bias_state = [np.abs(np.random.rand(batch_size, s)/100)	for s in state_size[0:-2]]

	one_hot = np.zeros((batch_size, state_size[-2]))
	one_hot[:,start_bias] = 1.
	bias_state.append(np.zeros((batch_size, state_size[-2])))

	one_hot = np.zeros((batch_size, state_size[-1]))
	one_hot[:,start_bias] = 1.
	bias_state.append(one_hot)

	return tuple(bias_state)


def get_training_batch(batch_size, seq_length, num_bits):

    bs = batch_size
    sl = seq_length
    nb = num_bits
    batch_y = np.zeros((bs, sl*2+1, nb+1))
    sequence = (np.random.rand(bs, sl, nb)*2).astype(int)
    
    batch_y[:,0:seq_length,0:nb] = sequence[:,:,:]
    batch_y[:,sl,num_bits] = 1
    batch_x = batch_y[:,:,:]
    batch_y[:,sl+1:,0:nb] = sequence[:,:,:]
    batch_y = batch_y[:,:,0:num_bits]

    #print(str(batch_x[0,0:2,:]))

    return batch_x, batch_y


M = 32
N = 15
S = 2
batch_size = 32
state_size = N*(M,) + (N, N)

session = tf.Session()

write = tf.Variable(tf.random_normal(dtype=tf.float64, shape=(batch_size, 3*M+S+3)))
read = tf.Variable(tf.random_normal(dtype=tf.float64, shape=(batch_size, M+S+3)))
# Split the forward portions of the read and write heads into 
# the various pieces. See paper by Alex Graves for more info.
write_pieces = tf.split(write, [M, M, M, S, 1, 1, 1], axis=1)
read_pieces = tf.split(read, [M, S, 1, 1, 1], axis=1)

write_keys = ['key', 'shift', 'beta', 'gamma', 'g', 'add', 'erase']
read_keys = ['key', 'shift', 'beta', 'gamma', 'g']

write_head = \
{
    #'key':tf.sigmoid(write_pieces[0]),
    'key':write_pieces[0],
    'add':tf.sigmoid(write_pieces[1]),
    'erase':tf.sigmoid(write_pieces[2]),
    'shift':tf.nn.softmax(write_pieces[3]),
    'beta':tf.nn.softplus(write_pieces[4]),
    'gamma':tf.nn.softplus(write_pieces[5]) + 1,
    'g':tf.sigmoid(write_pieces[6]),
}

read_head = \
{
    #'key':tf.sigmoid(read_pieces[0]),
    'key':read_pieces[0],
    'shift':tf.nn.softmax(read_pieces[1]),
    'beta':tf.nn.softplus(read_pieces[2]),
    'gamma':tf.nn.softplus(read_pieces[3]) + 1,
    'g':tf.sigmoid(read_pieces[4]),
}

cell_input = tf.concat([write_head[k] for k in write_keys] + \
    [read_head[k] for k in read_keys], axis=1)

output = head_ops(cell_input, bias_state(batch_size))
session.run(tf.global_variables_initializer())

for k in stufflist:
	print(k)
	print(session.run(stufflist[k]	))


print(session.run(1-write_head['g']))