from __future__ import print_function
import tensorflow as tf
import numpy as np
from ntm_cell import NTMCell
from ntm_cell import head_pieces_tuple_to_dict


np.set_printoptions(threshold=np.nan)

class NTM(object):
    '''
    Performs several operations relevant to the NTM:
      - builds the computation graph
      - trains the model
      - tests the model
    '''
    
    def __init__(self, mem_size, input_size, output_size, session,
                 num_heads=1, shift_range=3, name="NTM"):
        '''
        Builds the computation graph for the Neural Turing Machine.
        The tasks from the original paper call for the NTM to take in a
        sequence of arrays, and produce some output.
        Let B = batch size, T = sequence length, and L = array length, then
        a single input sequence is a matrix of size [TxL]. A batch of these
        input sequences has size [BxTxL].

        Arguments:
          mem_size - Tuple of integers corresponding to the number of storage
            locations and the dimension of each storage location (in the paper
            the memory matrix is NxM, mem_size refers to (N, M)).
          input_size - Integer number of elements in a single input vector
            (the value L).
          output_size - Integer number of elements in a single output vector.
          session - The TensorFlow session object that refers to the current
            computation graph.
          num_heads - The integer number of write heads the NTM uses (future
            feature).
          shift_range - The integer number of shift values that the read/write 
            heads can perform, which corresponds to the direction and magnitude
            of the allowable shifts.
            Shift ranges and corresponding available shift
            directions/magnitudes:
              3 => [-1, 0, 1]
              4 => [-2, -1, 0, 1] 
              5 => [-2, -1, 0, 1, 2]
          name - A string name for the variable scope, for troubleshooting.
        '''

        self.num_heads = 1
        self.sess = session
        self.S = shift_range
        self.N, self.M = mem_size
        self.in_size = input_size
        self.out_size = output_size

        num_lstm_units = 100
        self.dt=tf.float32

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

            self.ntm_cell = NTMCell(mem_size=(N, M), num_shifts=S)

            write_head, read_head = NTMCell.head_pieces(
                head_raw, mem_size=(N, M), num_shifts=S, axis=2)

            self.write_head, self.read_head = \
                head_pieces_tuple_to_dict(write_head, read_head)

            self.ntm_init_state = tuple(
                [tf.placeholder(dtype=dt, shape=(None, s)) \
                for s in self.ntm_cell.state_size])

            self.ntm_reads, self.ntm_last_state = tf.nn.dynamic_rnn(
                cell=self.ntm_cell, initial_state=self.ntm_init_state,
                inputs=head_raw, dtype=dt)

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
        '''
        Builds a single-layer LSTM controller that manipulates values in the 
        memory matrix and helps produce output. This method should only be 
        utilized by the class.

        Arguments:
          inputs - TF tensor containing data that is passed to the controller.
          batch_size - The number of sequences in a given training batch.
          seq_length - The length of the sequence being passed to the 
            controller.
          num_units - The number of units inside of the LSTM controller.
        '''

        N = self.N
        M = self.M
        S = self.S
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
            inputs=inputs, dtype=dt)

        lstm_out = tf.tanh(lstm_out_raw)
        lstm_out_flat = tf.reshape(lstm_out, [-1, num_units])

        # The number of nodes on the controller's output is determined by
        #   1. the number of allowable shifts
        #   2. the width of the columns in the memory matrix
        head_nodes = 4*M+2*S+6

        head_W = tf.Variable(
            tf.random_normal([num_units, num_heads*head_nodes]), name='head_W')
        head_b_W = tf.Variable(
            tf.random_normal([num_heads*head_nodes,]), name='head_b_W')

        head_raw_flat = tf.matmul(lstm_out_flat, head_W) + head_b_W
        head_raw = tf.reshape(head_raw_flat, [batch_size, seq_length, head_nodes])

        return head_raw

    def train_batch(self, batch_x, batch_y, learning_rate=1e-4):
        '''
        Trains the model on a batch of inputs and their corresponding outputs.
        Returns the error that was obtained by training the NTM on the input
        sequence that is provided as an argument.

        Arguments:
          batch_x - The batch of input training sequences [BxTxL1]. Note that
            the first two dimensions (batch size and sequence length) of both
            batches MUST be the same. Numpy array.
          batch_y - The batch of output training sequences [BxTxL2]. The 
            output sequences are the desired outputs after the NTM has been
            presented with the training input, batch_x. Numpy array.

        Outputs:
          error - The amount of error (float)produced from this particular 
            training sequence. The error operation is defined in the
            constructor.
        '''

        lr = learning_rate
        batch_size = batch_x.shape[0]
        ntm_init_state = self.ntm_cell.bias_state(batch_size)
        lstm_init_state = tuple([np.zeros((batch_size, s)) \
            for s in self.lstm_cell.state_size])

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
        '''
        Passes a single input sequence to the NTM, and produces an output
        according to what it's learned. Returns a tuple of items of interest
        for troubleshooting purposes (the read/write vectors and output).

        Arguments:
          test_x - A batch of input sequences [BxTxL1] that the NTM will use to
            produce a batch of output sequences [BxTxL2]. Numpy array.

        Outputs:
          output_b - A numpy array representing the output of the NTM after
            being presented with the input batch [BxTxL2].
          w_read_b - A numpy array of "read" locations that the NTM used.
            From the paper, write locations are normalized vectors that allow
            the NTM to focus on rows of the memory matrix.
          w_write_b - A numpy array of "write" locations that the NTM used.
          g_read_b - A numpy array of scalar values indicating whether the NTM
            used the previous read location or associative recall to determine
            the read location at each timestep.
          g_write_b - A numpy array of scalar values indicating whether the NTM
            used the previous write location or associative recall to determine
            the write location at each timestep.
          s_read_b - A numpy array of vectors describing the magnitude and
            direction of the shifting operation that was applied to the read
            head.
          s_write_b - A numpy array of vectors describing the magnitude and
            direction of the shifting operation that was applied to the write
            head.
        '''

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

