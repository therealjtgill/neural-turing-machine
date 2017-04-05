from __future__ import print_function
import tensorflow as tf
import numpy as np
from utils import *
from datetime import datetime
from ntm_ops import NTMCell
from time import time

np.set_printoptions(threshold=np.nan)

class NTM(object):
    def __init__(self, mem_size, session, vec_size, shift_range=2, name="NTM"):
        '''
        Currently not using a fully connected layer directly before or after
        the LSTM.

                                input
                                  |
                                 LSTM
                                 /  \
                  M_t-1, w_t-1  /    \   M_t-1, w_t-1
                            \  /      \   /
                             read     write
                              |         |
                        output, w_t    M_t, w_t
                                 |
                            dim_reduce
                                 |
                               output

        The read/write heads are going to have to be encased in their own
        object that inherits from RNNCell. Doing this will allow the serial
        operations of reading/writing to the matrix and creating the address
        vectors to be done with a dynamic number of timesteps.
        There might be a way to do this with TF's built-in control flow
        statements, but scaredface.jpg.
        '''
        self.session = session
        S = shift_range
        N, M = mem_size
        num_lstm_units=100
        num_head_units=100

        with tf.variable_scope(name):
            self.feed_x = tf.placeholder(dtype=tf.float32,
                shape=(None, None, vec_size))
            self.feed_y = tf.placeholder(dtype=tf.float32,
                shape=(None, None, vec_size-1))
            self.feed_lr = tf.placeholder(dtype=tf.float32, shape=())

            batch_size = tf.shape(self.feed_x)[0]
            num_instr = tf.shape(self.feed_x)[1]

            self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units= \
                num_lstm_units, forget_bias=1.0)

            lstm_init_state = tuple(
                [tf.placeholder(dtype=tf.float32, shape=(None, s)) \
                for s in self.lstm_cell.state_size])

            self.lstm_init_state = tf.contrib.rnn.LSTMStateTuple(
                lstm_init_state[0], lstm_init_state[1])

            self.lstm_outputs, self.last_lstm_state = tf.nn.dynamic_rnn( \
                cell=self.lstm_cell, initial_state=self.lstm_init_state,
                inputs=self.feed_x, dtype=tf.float32, parallel_iterations=64)

            self.lstm_outputs = tf.tanh(self.lstm_outputs)

            lstm_outputs_reshaped = tf.reshape(self.lstm_outputs,
                [-1, num_lstm_units])

            # Write head weights/biases
            self.J = tf.Variable(tf.random_normal([num_lstm_units, 3*M+S+3],
                stddev=0.01))
            self.b_J = tf.Variable(tf.random_normal([3*M+S+3,], stddev=0.01))

            # Read head weights/biases
            self.K = tf.Variable(tf.random_normal([num_lstm_units, M+3+S],
                stddev=0.01))
            self.b_K = tf.Variable(tf.random_normal([M+S+3,], stddev=0.01))

            self.write_raw = tf.matmul(lstm_outputs_reshaped, self.J) + self.b_J
            self.read_raw = tf.matmul(lstm_outputs_reshaped, self.K) + self.b_K

            self.write_raw = tf.reshape(self.write_raw,
                [batch_size, num_instr, 3*M+S+3])
            self.read_raw = tf.reshape(self.read_raw,
                [batch_size, num_instr, M+S+3])

            #cell_input = tf.concat([self.write_head[k] for k in write_keys] + \
            #    [self.read_head[k] for k in read_keys], axis=2)

            cell_input = tf.concat([self.write_raw, self.read_raw], axis=2)
            
            self.ntm_cell = NTMCell(mem_size=(N,M), shift_range=S)
            
            self.write_head, self.read_head = NTMCell.head_pieces(
                self.write_raw, self.read_raw, mem_size, S, 2, 'dict')

            self.ntm_init_state = tuple(
                [tf.placeholder(dtype=tf.float32, shape=(None, s)) \
                for s in self.ntm_cell.state_size])

            self.ntm_outputs, self.last_ntm_state = tf.nn.dynamic_rnn( \
                cell=self.ntm_cell, initial_state=self.ntm_init_state,
                inputs=cell_input, dtype=tf.float32, parallel_iterations=64)

            self.read_addresses = self.ntm_outputs[-2]
            self.write_addresses = self.ntm_outputs[-3]

            self.L = tf.Variable(tf.random_normal([M, vec_size-1], stddev=0.01))
            self.b_L = tf.Variable(tf.random_normal([vec_size-1,], stddev=0.01))

            #splits = [(num_instr-1)/2 + 1, num_instr - ((num_instr-1)/2 + 1)]
            read_values_flat = tf.reshape(self.ntm_outputs[-1], [-1,M])
            logits_flat = tf.matmul(read_values_flat, self.L) + self.b_L
            targets_flat = tf.reshape(self.feed_y, [-1,vec_size-1])

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=targets_flat, logits=logits_flat))

            self.predictions_flat = tf.sigmoid(logits_flat)

            self.optimizer = tf.train.RMSPropOptimizer(
                learning_rate=self.feed_lr, momentum=0.9)

            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            capped_grads = [(tf.clip_by_value(grad, -10., 10.), var) \
                for grad, var in grads_and_vars]

            self.train_op = self.optimizer.apply_gradients(capped_grads)
            #self.train_op = self.optimizer.minimize(self.loss)

    def train_batch(self, batch_x, batch_y, learning_rate=1e-4, get_ntm_outputs=False):
        '''
        Args:
            batch_x: Batch of instructions to be sent to the controller.
              (batch_size, num_instr, num_bits)
            batch_y: Batch of read results for each instruction sent to the
              controller.
              (batch_size, num_instr, num_bits)
            get_ntm_outputs: Boolean value indicating whether the outputs of
              the NTM should be returned along with the batch training error.

        Returns:
            error: The training error as a percentage of correctness.
            [outputs]: The full output of the NTM for each batch and 
              instruction sent to the NTM. This is only returned if the
              'get_ntm_outputs' flag is set to True.
        '''
        
        lr = learning_rate
        batch_size = batch_x.shape[0]
        ntm_init_state = self.ntm_cell.bias_state(batch_size)
        #lstm_init_state = self.lstm_cell.zero_state(batch_size)
        lstm_init_state = tuple(
            [np.zeros((batch_size, s)) for s in self.lstm_cell.state_size])
        fetches = [self.loss, self.train_op]
        feeds = {self.feed_x: batch_x, self.feed_y: batch_y, self.feed_lr:lr}
        
        for i in range(len(ntm_init_state)):
            feeds[self.ntm_init_state[i]] = ntm_init_state[i]

        for i in range(len(lstm_init_state)):
            feeds[self.lstm_init_state[i]] = lstm_init_state[i]

        if get_ntm_outputs:
            fetches.append(self.read_addresses)
            fetches.append(self.write_addresses)
            fetches.append(self.read_head)
            fetches.append(self.write_head)
            fetches.append(self.last_ntm_state)
            fetches.append(self.predictions_flat)
            error, _, ra, wa, rh, wh, state, pred,  = self.session.run(fetches, feeds)
            return error, ra, wa, rh, wh, state, pred
        else:
            error, _ = self.session.run(fetches, feeds)
            #print('random state:')
            return error

    def run_once(self, test_x):
    	'''
    	Grabs the read/write addresses and output from running the NTM with
    	'test_x' as the input. Currently the method only tests the first
    	set of instructions represented by batch_x.
        Args:
            test_x: Batch of instructions to be sent to the controller for
              testing.
              (batch_size, num_instr, num_bits)

        Returns:
            output_block: The full output of the NTM for each batch and 
              instruction sent to the NTM; a 2D numpy array.
            write_addresses_block: The full set of write addresses used in this
              sequence of tasks; a 2D numpy array.
            read_addresses_block: The full set of read addresses used in this
              sequence of tasks; a 2D numpy array.
        '''

        if (test_x.shape[0] < 2):
        	raise Exception('The batch size of the test input should be > 2.')

        batch_size = test_x.shape[0]
        num_seq = test_x.shape[1]
        sequences = np.split(test_x, num_seq, axis=1)
        ntm_init_state = self.ntm_cell.bias_state(batch_size)
        #lstm_init_state = self.lstm_cell.zero_state(batch_size, dtype=tf.float32)
        lstm_init_state = tuple(
            [np.zeros((batch_size, s)) for s in self.lstm_cell.state_size])

        outputs = []
        write_addresses = []
        read_addresses = []
        for seq in sequences:
            fetches = [self.predictions_flat, self.last_ntm_state,
                self.last_lstm_state]
            feeds = {self.feed_x: seq}

            for i in range(len(ntm_init_state)):
                feeds[self.ntm_init_state[i]] = ntm_init_state[i]

            for i in range(len(lstm_init_state)):
                feeds[self.lstm_init_state[i]] = lstm_init_state[i]

            output, ntm_init_state, lstm_init_state = \
                self.session.run(fetches, feeds)

            outputs.append(output[0])
            write_addresses.append(ntm_init_state[-2][0])
            read_addresses.append(ntm_init_state[-1][0])

        output_block = np.array(outputs)
        write_addresses_block = np.array(write_addresses)
        read_addresses_block = np.array(read_addresses)

        return output_block, write_addresses_block, read_addresses_block

def get_training_batch(batch_size, seq_length, num_bits):
	'''
	Returns batches of data for training or testing. For the copy task, the
	data returned is in the form:
	  input:   [pattern], [delimiter], [zeros]
	  targets: [zeros],   [zerp],      [pattern]
	Where 'pattern' is a sequence of num_bits binary values that the network
	is expected to internalize and replicate after seeing the delimiter value.

	Args:
		batch_size: The number of batches of sequences that the network should
		  be trained on (integer).
		seq_length: The length of the sequences of binary vectors that the 
		  network should reproduce.
		num_bits: The number of bits in each binary array.

	Returns:
		batch_x: Batch of sequences of binary vectors that the network will be
		  trained on.
		  [batch_size, seq_length*2 + 1, num_bits + 1]
		batch_y: Batch of sequences of binary vectors that the network should
		  produce after being presented with batch_x.
		  [batch_size, seq_length*2 + 1, num_bits]
	'''

    bs = batch_size
    sl = seq_length
    nb = num_bits
    batch_x = np.zeros((bs, sl*2+1, nb+1))
    batch_y = np.zeros((bs, sl*2+1, nb))
    sequence = (np.random.rand(bs, sl, nb)*2).astype(int)
    
    batch_x[:,0:sl,0:nb] = sequence[:,:,:].copy()
    #batch_y[:,0:sl,0:nb] = sequence[:,:,:].copy()
    batch_y[:,sl+1:2*sl+1,0:nb] = sequence[:,:,:].copy()
    batch_x[:,sl,num_bits] = 1.
    batch_x[:,sl+1:sl*2+1,:] = 0.

    return batch_x, batch_y

def main():
    #print("in main")
    #get_training_batch(32, 10, 8)
    #np.set_printoptions(threshold='nan')
    prev_time = time()
    shape=(32,15)
    session = tf.Session()
    batch_size = 64
    train = True
    load = not train
    date = datetime.now()
    date = str(date).replace(' ', '').replace(':', '-')
    save_dir = make_dir(date)
    print_thresh = 100
    avg_error = 0
    prev_vals = []
    
    ntm = NTM(shape, session, 9, shift_range=3)
    session.run(tf.global_variables_initializer())
    print('graph built')

    saver = tf.train.Saver(tf.global_variables())
    lr = 1e-4

    if train:
        #saver.restore(session, "models/2017-04-02073947.281558ntm.ckpt")
        for step in range(50000):
            num_instr = int(np.random.rand()*12)+8
            #num_instr = 10
            batch_x, batch_y = get_training_batch(batch_size, num_instr, 8)
            
            if step % print_thresh == 0:
                time_elapsed = time() - prev_time
                prev_time = time()
                print('-------------------------------------------')
                print('step:', step)
                #print('time elapsed:', time_elapsed)

                error, ra, wa, rh, wh, state, pred = ntm.train_batch(batch_x, 
                    batch_y, lr, True)
                avg_error += error

                sample_instr = int(np.random.rand()*num_instr)
                
                ultra_print(error, ra, wa, rh, wh, state, pred, batch_y,
                    sample_instr, batch_size, num_instr)
                prev_vals=list([error, ra, wa, rh, wh, state, pred, batch_y])
                    
                print('loop train error:', step, error)
                print('average error:', avg_error/print_thresh)

                save_error(avg_error, save_dir, date)

                if np.isnan(error):
                    exit()
                avg_error = 0
            else:
                error, ra, wa, rh, wh, state, pred = ntm.train_batch(batch_x,
                	batch_y, lr, True)
                avg_error += error
                print('step:', step, error, 'sequence length:', num_instr)

                if np.isnan(error):
                    sample_instr = int(np.random.rand()*num_instr)
                    
                    ultra_print(error, ra, wa, rh, wh, state, pred, batch_y,
                        sample_instr, batch_size, num_instr)
                    error, ra, wa, rh, wh, state, pred, batch_y = tuple(prev_vals[0:-2])
                    num_instr=prev_vals[-1]
                    batch_x=prev_vals[-2]
                    ultra_print(error, ra, wa, rh, wh, state, pred, batch_y,
                        sample_instr, batch_size, num_instr)
                    print('minimum predicted value:', np.min(pred))
                    exit()

                prev_vals=list([error, ra, wa, rh, wh, state, pred, batch_y,
                    batch_x, num_instr])

            if step % 500 == 0:
                test_run(ntm, save_dir, step)

            if step % 500 == 0 and step != 0:
                saver.save(session, os.path.join(save_dir, "model.ntm.ckpt"))
                print('Model saved!')
    else:
        #saver.restore(session, "models/" + date + "ntm.ckpt")
        test_run(ntm, save_dir, step)

def test_run(ntm, folder, step):
    num_instr = (10, 20, 30)
    
    for n in num_instr:
        test_x, test_y = get_training_batch(2, n, 8)

        pred, w_add, r_add = ntm.run_once(test_x)
        #pred = np.reshape(pred, [2, n*2 + 1, -1])

        suffix = str(n) + '-' + str(step)

        save_output_plot(test_y, pred, folder, 'output' + suffix)
        save_address_plot(w_add, folder, 'writeadd' + suffix)
        save_address_plot(r_add, folder, 'readadd' + suffix)


def ultra_print(error, ra, wa, rh, wh, state, pred, batch_y, sample_instr, batch_size, num_instr):
    print('read addresses:\n', str(ra[sample_instr,-5:-1,:]))
    #print('read head key:\n', rh['key'][sample_instr,0:5,:])
    #print('read head shift:\n', rh['shift'][sample_instr,0:5,:])
    #print('read head g:\n', rh['g'][sample_instr,0:5,:])
    print('read head gamma:\n', np.max(rh['gamma']))
    print('\t----')    
    print('write addresses:\n', str(wa[sample_instr,-5:-1,:]))
    #print('write head key:\n', wh['key'][sample_instr,0:5,:])
    #print('write head shift:\n', wh['shift'][sample_instr,0:5,:])
    #print('write head g:\n', wh['g'][sample_instr,0:5,:])
    print('write head gamma:\n', np.max(wh['gamma']))
    #print('predictions:\n', np.reshape(pred, 
    #    [batch_size, num_instr*2+1, -1])[0,-5:-1,:])
    print('predictions:\n', np.reshape(pred, 
        [batch_size, num_instr*2+1, -1])[0,-5:-1,:])
    #print('targets:\n', np.reshape(batch_y, 
    #    [batch_size, num_instr*2+1, -1])[0,-5:-1,:])
    print('targets:\n', np.reshape(batch_y, 
        [batch_size, num_instr*2+1, -1])[0,-5:-1,:])

if __name__ == "__main__":
    main()