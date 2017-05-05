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

            # Started conversion to the multi-head output here, still have
            # lots to do.
            self.w_read = self.ntm_last_state[N:N+num_heads]
            self.w_write = self.ntm_last_state[N+num_heads:N+2*num_heads]

            ntm_reads_flat = [tf.reshape(r, [-1, M]) for r in self.ntm_reads]

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
                read_head, write_head = self.sess.run(fetches, feeds)

            outputs.append(output[0].copy())
            w_read.append(ntm_init_state[-2][0].copy())
            w_write.append(ntm_init_state[-1][0].copy())
            g_read.append(read_head['g'][0,0,:].copy())
            g_write.append(write_head['g'][0,0,:].copy())
            s_read.append(read_head['shift'][0,0,:].copy())
            s_write.append(write_head['shift'][0,0,:].copy())

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

