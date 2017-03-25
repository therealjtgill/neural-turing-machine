from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
            self.feed_x = tf.placeholder(dtype=tf.float32, shape=(None, None, vec_size))
            self.feed_y = tf.placeholder(dtype=tf.float32, shape=(None, None, vec_size-1))

            batch_size = tf.shape(self.feed_x)[0]
            num_instr = tf.shape(self.feed_x)[1]

            self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units= \
                num_lstm_units, forget_bias=1.0)

            self.lstm_outputs, self.last_hidden_state = tf.nn.dynamic_rnn(cell= \
                self.lstm_cell, inputs=self.feed_x, dtype=tf.float32)

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

            self.write_raw = tf.reshape(self.write_raw, [batch_size, num_instr, 3*M+S+3])
            self.read_raw = tf.reshape(self.read_raw, [batch_size, num_instr, M+S+3])

            # Split the forward portions of the read and write heads into 
            # the various pieces. See paper by Alex Graves for more info.
            write_pieces = tf.split(self.write_raw, [M, M, M, S, 1, 1, 1], axis=2)
            read_pieces = tf.split(self.read_raw, [M, S, 1, 1, 1], axis=2)

            write_keys = ['key', 'shift', 'beta', 'gamma', 'g', 'add', 'erase']
            read_keys = ['key', 'shift', 'beta', 'gamma', 'g']

            self.write_head = \
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

            self.read_head = \
            {
                #'key':tf.sigmoid(read_pieces[0]),
                'key':read_pieces[0],
                'shift':tf.nn.softmax(read_pieces[1]),
                'beta':tf.nn.softplus(read_pieces[2]),
                'gamma':tf.nn.softplus(read_pieces[3]) + 1,
                'g':tf.sigmoid(read_pieces[4]),
            }

            #cell_input = tf.concat([self.write_head[k] for k in write_keys] + \
            #    [self.read_head[k] for k in read_keys], axis=2)

            cell_input = tf.concat([self.write_raw, self.read_raw], axis=2)
            #print(cell_input)
            self.ntm_cell = NTMCell(mem_size=(N,M), shift_range=S)
            #ntm_state = ntm_cell.bias_state(batch_size)
            #print('init state:', self.ntm_init_state)

            self.ntm_init_state = tuple(
                [tf.placeholder(dtype=tf.float32, shape=(None, s)) \
                for s in self.ntm_cell.state_size])

            self.ntm_outputs, self.last_ntm_states = tf.nn.dynamic_rnn( \
                cell=self.ntm_cell, initial_state=self.ntm_init_state,
                inputs=cell_input, dtype=tf.float32)
            self.read_addresses = self.ntm_outputs[-2]
            self.write_addresses = self.ntm_outputs[-3]

            self.L = tf.Variable(tf.random_normal([M, vec_size-1], stddev=0.01))
            self.b_L = tf.Variable(tf.random_normal([vec_size-1,], stddev=0.01))
            #print('ntm_outputs:', self.ntm_outputs)

            read_values_flat = tf.reshape(self.ntm_outputs[-1], [-1,M])
            logits_flat = tf.matmul(read_values_flat, self.L) + self.b_L
            #predictions_flat = tf.sigmoid(predictions_flat)

            #tf.clip_by_value(J, -10., 10.)

            #logits_flat = tf.reshape(logits, [-1,M])
            targets_flat = tf.reshape(self.feed_y, [-1,vec_size-1])
            #self.loss = tf.reduce_mean(tf.losses.sigmoid_cross_entropy( \
            #    multi_class_labels=feed_y_flat, logits=logits_flat))

            self.loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=targets_flat, logits=logits_flat))

            self.predictions_flat = tf.sigmoid(logits_flat)
            #self.loss = tf.reduce_mean(
            #    -1*targets_flat*tf.log(predictions_flat) - \
            #    (1-targets_flat)*tf.log(1-predictions_flat))

            self.optimizer = tf.train.RMSPropOptimizer(
                learning_rate=0.001, momentum=0.9)

            grads_and_vars = self.optimizer.compute_gradients(self.loss)
            capped_grads = [(tf.clip_by_value(grad, -10., 10.), var) \
                for grad, var in grads_and_vars]

            self.train_op = self.optimizer.apply_gradients(capped_grads)

    def train(self, batch_x, batch_y, get_ntm_outputs=False):
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
        
        batch_size = batch_x.shape[0]
        init_state = self.ntm_cell.bias_state(batch_size)
        fetches = [self.loss, self.train_op]
        feeds = {self.feed_x: batch_x, self.feed_y: batch_y}

        for i in range(len(init_state)):
            feeds[self.ntm_init_state[i]] = init_state[i]

        if get_ntm_outputs:
            fetches.append(self.read_addresses)
            fetches.append(self.write_addresses)
            fetches.append(self.read_head)
            fetches.append(self.write_head)
            fetches.append(self.last_ntm_states)
            fetches.append(self.predictions_flat)
            error, _, ra, wa, rh, wh, state, pred,  = self.session.run(fetches, feeds)
            return error, ra, wa, rh, wh, state, pred
        else:
            error, _ = self.session.run(fetches, feeds)
            #print('random state:')
            return error
        
    def run(self, test_x, get_ntm_outputs=False):
        '''
        Args:
            test_x: Batch of instructions to be sent to the controller for
              testing.
              (batch_size, num_instr, num_bits)
            get_ntm_outputs: Boolean value indicating whether the outputs of
              the NTM should be returned along with the batch training error.

        Returns:
            outputs: The full output of the NTM for each batch and instruction
              sent to the NTM.
        '''
        batch_size = test_x.shape[0]
        init_state = self.ntm_cell.bias_state(batch_size)
        fetches = self.predictions_flat
        feeds = {self.feed_x: test_x}

        for i in range(len(init_state)):
            feeds[self.ntm_init_state[i]] = init_state[i]

        outputs = self.session.run(fetches, feeds)

        return outputs

def get_training_batch(batch_size, seq_length, num_bits):

    bs = batch_size
    sl = seq_length
    nb = num_bits
    batch_x = np.zeros((bs, sl*2+1, nb+1))
    batch_y = np.zeros((bs, sl*2+1, nb))
    sequence = (np.random.rand(bs, sl, nb)*2).astype(int)
    
    batch_x[:,0:sl,0:nb] = sequence[:,:,:]
    batch_y[:,0:sl,0:nb] = sequence[:,:,:]
    batch_y[:,sl+1:2*sl+1,0:nb] = sequence[:,:,:]
    batch_x[:,sl,num_bits] = 1
    #batch_x = batch_y[:,:,:]
    #batch_y[:,sl+1:,0:nb] = sequence[:,:,:]
    #batch_y = batch_y[:,:,0:num_bits]
    batch_x[:,sl+1:sl*2+1,:] = 0

    #print(str(batch_x[0,0:2,:]))

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
    date = str(date).replace(' ', '').replace(':', '')
    print_thresh = 100
    avg_error = 0
    prev_vals = []

    #test_x, test_y = get_training_batch(2, 20, 8)
    #plt.imshow(test_x[0].T)
    #plt.show()
    #plt.imshow(test_y[0].T)
    #plt.show()
    
    ntm = NTM(shape, session, 9, 3)
    session.run(tf.global_variables_initializer())
    print('graph built')

    saver = tf.train.Saver(tf.global_variables())

    if train:
        #saver.restore(session, "models/2017-03-22193204.482822ntm.ckpt")
        for step in range(50000):
            num_instr = int(np.random.rand()*12)+8
            #num_instr = 10
            batch_x, batch_y = get_training_batch(batch_size, num_instr, 8)
            
            if step % print_thresh == 0:
                time_elapsed = time() - prev_time
                prev_time = time()
                print('-------------------------------------------')
                print('step:', step)
                print('time elapsed:', time_elapsed)
                error, ra, wa, rh, wh, state, pred = ntm.train(batch_x, batch_y, True)
                avg_error += error

                sample_instr = int(np.random.rand()*num_instr)
                
                ultra_print(error, ra, wa, rh, wh, state, pred, batch_y,
                    sample_instr, batch_size, num_instr)
                prev_vals=list([error, ra, wa, rh, wh, state, pred, batch_y])
                    
                print('loop train error:', step, error)
                print('average error:', avg_error/print_thresh)
                avg_error = 0
                if np.isnan(error):
                    exit()
            else:
                error, ra, wa, rh, wh, state, pred = ntm.train(batch_x, batch_y, True)
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
                    print('full desired output:', batch_y)
                    exit()

                prev_vals=list([error, ra, wa, rh, wh, state, pred, batch_y,
                    batch_x, num_instr])

            if step % 1000 == 0 and step != 0:
                saver.save(session, date + "ntm.ckpt")
                print('Model saved!')

        saver.save(session, date + "ntm.ckpt")
    else:
        
        #saver.restore(session, "models/" + date + "ntm.ckpt")
        num_instr = (20, 30, 50, 100)

        test_x, test_y = get_training_batch(2, 20, 8)
        plt.imshow(test_x[0].T)
        plt.show()
        plt.imshow(test_y[0].T)
        plt.show()
        print('finished')
        '''
        for n in num_instr:
            test_x, test_y = get_training_batch(2, n, 8)
            plt.imshow(test_x[0].T)
            plt.show()
            print('finished')

            pred = ntm.run(test_x)
            pred = np.reshape(pred, [2, n*2+1, -1])
            print(pred.shape)
            plt.imshow(pred[0].T)
            plt.show()
        '''

def ultra_print(error, ra, wa, rh, wh, state, pred, batch_y, sample_instr, batch_size, num_instr):
    print('read addresses:\n', str(ra[sample_instr,0:5,:]))
    print('read head key:\n', rh['key'][sample_instr,0:5,:])
    print('read head shift:\n', rh['shift'][sample_instr,0:5,:])
    print('read head g:\n', rh['g'][sample_instr,0:5,:])
    print('read head gamma:\n', rh['gamma'][sample_instr,0:5,:])
    print('\t----')    
    print('write addresses:\n', str(wa[sample_instr,0:5,:]))
    print('write head key:\n', wh['key'][sample_instr,0:5,:])
    print('write head shift:\n', wh['shift'][sample_instr,0:5,:])
    print('write head g:\n', wh['g'][sample_instr,0:5,:])
    print('write head gamma:\n', wh['gamma'][sample_instr,0:5,:])
    print('predictions:\n', np.reshape(pred, 
        [batch_size, num_instr*2+1, -1])[0,0:5,:])
    print('targets:\n', np.reshape(batch_y, 
        [batch_size, num_instr*2+1, -1])[0,0:5,:])

if __name__ == "__main__":
    main()