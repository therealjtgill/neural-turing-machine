from __future__ import print_function
import tensorflow as tf
import numpy as np
from utils import *
from datetime import datetime
from ntm_cell import NTMCell
from time import time

np.set_printoptions(threshold=np.nan)

class NTM(object):

    def __init__(self, mem_size, input_size, output_size, session,
                 num_heads=1, shift_range=3, name="NTM"):

        self.num_heads = 1
        self.sess = session
        self.S = shift_range
        self.N, self.M = mem_size
        self.in_size = input_size
        self.out_size = output_size

        num_lstm_units = 100
        self.dt=tf.float32
        self.pi = 64

        pi = self.pi
        dt = self.dt
        N = self.N
        M = self.M
        S = self.S
        num_heads = self.num_heads

        with tf.variable_scope(name):
            self.feed_in = tf.placeholder(dtype=dt,
                shape=(None, None, input_size))

            self.feed_out = tf.placeholder(dtype=dt,
                shape=(None, None, output_size))

            self.feed_learning_rate = tf.placeholder(dtype=dt, 
                shape=())

            batch_size = tf.shape(self.feed_in)[0]
            seq_length = tf.shape(self.feed_in)[1]

            head_raw = self.controller(self.feed_in, batch_size, seq_length)

            self.ntm_cell = NTMCell(mem_size=(N, M), shift_range=S)

            self.write_head, self.read_head = NTMCell.head_pieces(
                head_raw, mem_size=(N, M), shift_range=S, axis=2, style='dict')

            self.ntm_init_state = tuple(
                [tf.placeholder(dtype=dt, shape=(None, s)) \
                for s in self.ntm_cell.state_size])

            self.ntm_reads, self.ntm_last_state = tf.nn.dynamic_rnn(
                cell=self.ntm_cell, initial_state=self.ntm_init_state,
                inputs=head_raw, dtype=dt, parallel_iterations=pi)

            self.w_read = self.ntm_last_state[-2]
            self.w_write = self.ntm_last_state[-1]

            ntm_reads_flat = tf.reshape(self.ntm_reads, [-1, M])

            L = tf.Variable(tf.random_normal([M, output_size]))
            b_L = tf.Variable(tf.random_normal([output_size,]))

            logits_flat = tf.matmul(ntm_reads_flat, L) + b_L
            targets_flat = tf.reshape(self.feed_out, [-1, output_size])

            self.error = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=targets_flat, logits=logits_flat))

            self.predictions = tf.sigmoid(
                tf.reshape(logits_flat, [batch_size, seq_length, output_size]))

            optimizer = tf.train.RMSPropOptimizer(
                learning_rate=self.feed_learning_rate, momentum=0.9)

            grads_and_vars = optimizer.compute_gradients(self.error)
            capped_grads = [(tf.clip_by_value(grad, -10., 10.), var) \
                for grad, var in grads_and_vars]

            self.train_op = optimizer.apply_gradients(capped_grads)

    def controller(self, inputs, batch_size, seq_length, num_units=100):
        N = self.N
        M = self.M
        S = self.S
        pi = self.pi
        dt = self.dt
        num_heads = self.num_heads

        self.lstm_cell = tf.contrib.rnn.BasicLSTMCell(
            num_units=num_units, forget_bias=1.0)

        self.lstm_init_state = tuple(
            [tf.placeholder(dtype=dt, shape=(None, s))
            for s in self.lstm_cell.state_size])

        lstm_init_state = tf.contrib.rnn.LSTMStateTuple(
            self.lstm_init_state[0], self.lstm_init_state[1])

        lstm_out_raw, self.lstm_last_state = tf.nn.dynamic_rnn(
            cell=self.lstm_cell, initial_state=lstm_init_state,
            inputs=inputs, dtype=dt, parallel_iterations=pi)

        lstm_out = tf.tanh(lstm_out_raw)
        lstm_out_flat = tf.reshape(lstm_out, [-1, num_units])

        head_nodes = 4*M+2*S+6

        head_W = tf.Variable(
            tf.random_normal([num_units, num_heads*head_nodes]), name='head_W')
        head_b_W = tf.Variable(
            tf.random_normal([num_heads*head_nodes,]), name='head_b_W')

        head_raw_flat = tf.matmul(lstm_out_flat, head_W) + head_b_W
        head_raw = tf.reshape(head_raw_flat, [batch_size, seq_length, head_nodes])

        return head_raw

    def train_batch(self, batch_x, batch_y, learning_rate=1e-4):

        lr = learning_rate
        batch_size = batch_x.shape[0]
        ntm_init_state = self.ntm_cell.bias_state(batch_size)
        lstm_init_state = tuple(
            [np.zeros((batch_size, s)) for s in self.lstm_cell.state_size])

        fetches = [self.error, self.train_op]
        feeds = {
            self.feed_in:batch_x,
            self.feed_out:batch_y,
            self.feed_learning_rate:lr
        }

        for i in range(len(ntm_init_state)):
            feeds[self.ntm_init_state[i]] = ntm_init_state[i]

        for i in range(len(lstm_init_state)):
            feeds[self.lstm_init_state[i]] = lstm_init_state[i]

        error, _ = self.sess.run(fetches, feeds)

        return error

    def run_once(self, test_x):
        batch_size = test_x.shape[0]
        num_seq = test_x.shape[1]
        sequences = np.split(test_x, num_seq, axis=1)
        ntm_init_state = self.ntm_cell.bias_state(batch_size)
        lstm_init_state = tuple(
            [np.zeros((batch_size, s)) for s in self.lstm_cell.state_size])

        outputs = []
        w_read = []
        w_write = []
        g_read = []
        g_write = []
        s_read = []
        s_write = []

        for seq in sequences:
            fetches = [self.predictions, self.ntm_last_state,
                self.lstm_last_state, self.read_head, self.write_head]
            feeds = {self.feed_in: seq}

            for i in range(len(ntm_init_state)):
                feeds[self.ntm_init_state[i]] = ntm_init_state[i]

            for i in range(len(lstm_init_state)):
                feeds[self.lstm_init_state[i]] = lstm_init_state[i]

            output, ntm_init_state, lstm_init_state, \
                write_head, read_head = self.sess.run(fetches, feeds)

            outputs.append(output[0])
            w_read.append(ntm_init_state[-2][0])
            w_write.append(ntm_init_state[-1][0])
            g_read.append(read_head['g'][0,0,:])
            g_write.append(write_head['g'][0,0,:])
            s_read.append(read_head['shift'][0,0,:])
            s_write.append(write_head['shift'][0,0,:])

        output_b = np.squeeze(np.array(outputs))
        w_read_b = np.array(w_read)
        w_write_b = np.array(w_write)
        g_read_b = np.array(g_read)
        g_write_b = np.array(g_write)
        s_read_b = np.array(s_read)
        s_write_b = np.array(s_write)

        #print(output_b.shape)

        return output_b, w_read_b, w_write_b, g_read_b, \
            g_write_b, s_read_b, s_write_b

def copy_task_batch(batch_size, seq_length, num_bits):
 
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
    
    mem_shape=(32,15)
    batch_size = 64
    avg_error = 0
    lr = 1e-4
    input_size = 9
    output_size = 8
    max_batches = 50000
    print_threshold = 100
    save_threshold = 500
    sequence_err = [0.,]*20
    sequences = [0.,]*20

    session = tf.Session()
    ntm = NTM(mem_shape, input_size, output_size, session)
    session.run(tf.global_variables_initializer())
    print('Computation graph built.')

    #train_model(ntm, batch_size=64, max_batches=5e4, save_threshold=100)

    date = datetime.now()
    date = str(date).replace(' ', '').replace(':', '-')
    save_dir = make_dir(date)
    saver = tf.train.Saver(tf.global_variables())

    for step in range(max_batches):
        seq_length = 8 + int(np.random.rand()*12)
        batch_in, batch_out = copy_task_batch(batch_size,
            seq_length, output_size)

        error = ntm.train_batch(batch_in, batch_out)
        avg_error += error/print_threshold

        if step % print_threshold == 0:
            for i in range(len(sequences)):
                if sequences[i] > 0.:
                    sequence_err[i] = sequence_err[i]/sequences[i]

            print('step: {0} average error: {1}'.format(step, avg_error))
            print('average sequence errors:')
            for i in range(len(sequence_err)):
                if sequence_err[i] > 0.:
                    print('sequence length:', i, 'average err:', sequence_err[i])
            sequence_err = [0.,]*20
            sequences = [0.,]*20
            save_text(error, save_dir, '0-train')
            avg_error = 0.
        else:
            sequence_err[seq_length] += error
            sequences[seq_length] += 1.
            
        if step % save_threshold == 0:
            saver.save(session, os.path.join(save_dir, "model.ntm.ckpt"))
            seq_length = (10, 20, 30)

            for s in seq_length:
                test_x, test_y = copy_task_batch(2, s, output_size)

                pred, w_r, w_w, g_r, g_w, s_r, s_w = ntm.run_once(test_x)

                suffix = str(s) + '-' + str(step)

                save_double_plot(test_y, pred, save_dir, 'output' + suffix,
                    'target', 'prediction')
                write_head_plot = (w_w, g_w, s_w)
                read_head_plot = (w_r, g_r, s_r)
                labels = ('address', 'g', 'shift')
                #save_single_plot(w_w, save_dir, 'writeadd' + suffix,
                #    'write address')
                #save_single_plot(w_r, save_dir, 'readadd' + suffix,
                #    'read address')
                #save_single_plot(g_r, save_dir, 'readforget' + suffix, '')
                #save_single_plot(g_w, save_dir, 'writeforget' + suffix, '')
                #save_single_plot(s_r, save_dir, 'readshift' + suffix, '')
                #save_single_plot(s_w, save_dir, 'writeshift' + suffix, '')
                save_multi_plot(write_head_plot, save_dir,
                    'writehead' + suffix, labels)
                save_multi_plot(read_head_plot, save_dir,
                    'readhead' + suffix, labels)



if __name__ == '__main__':
    main()