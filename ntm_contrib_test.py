'''
Unit test for Neural Turing Machine contribution.
'''

import tensorflow as tf
from tensorflow.python.platform import test
import numpy as np

from ntm_cell import NTMCell
from ntm_cell import address_regression
from ntm_np_forward import numpy_forward_pass
from ntm_np_forward import head_pieces
from ntm_np_forward import generate_address


class NTMRegression(object):
  '''
  A class that makes regression testing on the NTMCell easier
  '''
  def __init__(self, mem_size, session, num_heads=1, shift_range=3,
               name="NTM"):
    '''
    Just sets up an NTM without the controller. So all this will do is
    apply the NTM operations to some set of fake input.
    '''

    self.mem_size = mem_size
    self.shift_range = shift_range
    self.sess = session
    self.num_heads = num_heads

    (_, num_bits) = self.mem_size
    dt = tf.float32

    head_size = 4*num_bits + 2*self.shift_range + 6

    with tf.variable_scope(name):

      self.ntm_cell = NTMCell(mem_size=self.mem_size,
                              num_shifts=self.shift_range)

      # [batch_size, sequence_length, 4*M + 2*S + 6]
      self.feed_controller_input = \
        tf.placeholder(dtype=dt,
                       shape=(None, None, head_size))

      # ([batch_size, ntm_cell.state_size[0]], ...)
      self.feed_initial_state = \
        tuple([tf.placeholder(dtype=dt, shape=(None, s))
               for s in self.ntm_cell.state_size])

      self.ntm_reads, self.ntm_last_state = \
        tf.nn.dynamic_rnn(cell=self.ntm_cell,
                          initial_state=self.feed_initial_state,
                          inputs=self.feed_controller_input, dtype=dt)

      self.write_head, self.read_head = \
        self.ntm_cell.head_pieces(self.feed_controller_input,
                                  mem_size=self.mem_size,
                                  num_shifts=self.shift_range, axis=2)


  def run(self, controller_input, initial_state):
    '''
    Takes some controller input and initial state and spits out read/write
    addresses and values that are read from a memory matrix.
    '''

    (_, seq_length, _) = controller_input.shape
    sequences = np.split(controller_input, seq_length, axis=1)
    init_state = initial_state

    read_addresses = []
    write_addresses = []
    sequence_reads = []

    for seq in sequences:
      fetches = [self.ntm_reads, self.ntm_last_state]
      feeds = {self.feed_controller_input: seq}

      for i in range(len(init_state)):
        feeds[self.feed_initial_state[i]] = init_state[i]

      reads, last_state = self.sess.run(fetches, feeds)

      sequence_reads.append(reads)
      read_addresses.append(last_state[-2])
      write_addresses.append(last_state[-1])

      init_state = last_state

    read_addresses = \
      np.transpose(np.squeeze(np.array(read_addresses)), [1, 0, 2])
    write_addresses = \
      np.transpose(np.squeeze(np.array(write_addresses)), [1, 0, 2])
    sequence_reads = \
      np.transpose(np.squeeze(np.array(sequence_reads)), [1, 0, 2])

    return read_addresses, write_addresses, sequence_reads


class NTMForwardPassTest(test.TestCase):
  '''
  The NP and TF implementations are seeded with random initial values, the
  two implementations are fed the same mock controller output, and the
  results from each implementation are compared to each other.
  '''

  def setUp(self):
    '''
    Define the parameters that will be used to create the NP forward pass,
    then perform the NP forward pass.
    '''

    # Parameter definitions
    min_addresses = 5
    max_addresses = 10
    min_bits_per_address = 6
    max_bits_per_address = 12
    max_batch_size = 32
    min_batch_size = 10

    self.N = np.random.randint(low=min_addresses, high=max_addresses + 1)
    self.M = np.random.randint(low=min_bits_per_address,
                               high=max_bits_per_address + 1)
    #self.N, self.M = (10, 9)
    self.mem_size = (self.N, self.M)

    min_shifts = 3
    max_shifts = self.N - 1

    self.S = np.random.randint(low=min_shifts, high=max_shifts + 1)
    self.shift_range = self.S
    self.batch_size = np.random.randint(low=min_batch_size,
                                        high=max_batch_size)
    self.sequence_length = np.random.randint(low=3, high=max_addresses)
    #self.S, self.batch_size, self.sequence_length = (3, 12, 15)

    self.initial_state = NTMCell(self.mem_size,
                                 self.shift_range).bias_state(self.batch_size)

    self.controller_output = 10*np.random.rand(self.batch_size,
                                               self.sequence_length,
                                               4*self.M + 2*self.S + 6) - 5

    # Get the reference NP output for a single sequence (only one of the
    # batch items gets processed to completion).
    seq_initial_state = tuple([x[0, :] for x in self.initial_state])

    self.np_read_addresses, self.np_write_addresses, self.np_reads = \
      numpy_forward_pass(self.N,
                         self.M,
                         self.S,
                         seq_initial_state,
                         self.controller_output[0, :, :])


  def testTFAgainstNP(self):
    '''
    Compare the output of the Numpy implementation to the TensorFlow
    implementation given some random controller output and initial state.
    '''

    with self.test_session() as session:
      mem_size = (self.N, self.M)
      tf_regression_ntm = NTMRegression(mem_size, session, shift_range=self.S)

      # The tf_regression_ntm is an implementation of the NTMCell that captures
      # all of the addresses and reads and returns them as a bundle.
      output = tf_regression_ntm.run(self.controller_output, self.initial_state)

      # Take the first batch from the addresses and reads.
      tf_read_addresses = output[0][0, :, :]
      tf_write_addresses = output[1][0, :, :]
      tf_reads = output[2][0, :, :]

      # Convert matrices to lists of arrays to perform assertArrayNear
      # comparison.
      tf_read_addresses_list = np.split(tf_read_addresses,
                                        self.sequence_length)
      tf_write_addresses_list = np.split(tf_write_addresses,
                                         self.sequence_length)
      tf_reads_list = np.split(tf_reads, self.sequence_length)

      np_read_addresses_list = np.split(self.np_read_addresses,
                                        self.sequence_length)
      np_write_addresses_list = np.split(self.np_write_addresses,
                                         self.sequence_length)
      np_reads_list = np.split(self.np_reads, self.sequence_length)

      tf_read_addresses_list = [np.squeeze(item)
                                for item in tf_read_addresses_list]
      tf_write_addresses_list = [np.squeeze(item)
                                 for item in tf_write_addresses_list]
      tf_reads_list = [np.squeeze(item) for item in tf_reads_list]

      np_read_addresses_list = [np.squeeze(item)
                                for item in np_read_addresses_list]
      np_write_addresses_list = [np.squeeze(item)
                                 for item in np_write_addresses_list]
      np_reads_list = [np.squeeze(item) for item in np_reads_list]

      self.assertEqual(len(tf_read_addresses_list),
                       len(np_read_addresses_list))

      self.assertEqual(len(tf_write_addresses_list),
                       len(np_write_addresses_list))

      self.assertEqual(len(tf_reads_list),
                       len(np_reads_list))

      for i in range(self.sequence_length):
        self.assertArrayNear(tf_read_addresses_list[i],
                             np_read_addresses_list[i],
                             err=1e-5)

        self.assertArrayNear(tf_write_addresses_list[i],
                             np_write_addresses_list[i],
                             err=1e-5)

        self.assertArrayNear(tf_reads_list[i],
                             np_reads_list[i],
                             err=1e-5)


  def testHeadPieces(self):
    '''
    Show that the values extracted from the controller (key, gate, shift, etc.)
    are correct.
    '''

    mem_size = (self.N, self.M)
    np_read_head, np_write_head = head_pieces(self.controller_output,
                                              mem_size,
                                              self.S)

    with self.test_session() as session:
      tf_write_head, tf_read_head = NTMCell.head_pieces(self.controller_output,
                                                        mem_size,
                                                        self.S,
                                                        axis=2)
      tf_write_head, tf_read_head = session.run([tf_write_head, tf_read_head])

      # Make sure we got the same number of items from the read and write
      # heads.
      self.assertEqual(len(tf_write_head), len(np_write_head))
      self.assertEqual(len(tf_read_head), len(np_read_head))

      # Verify that the NP and TF read heads have approximately the same
      # values.
      for i in range(len(np_read_head)):
        for j in range(np_read_head[i].shape[0]):
          for k in range(np_read_head[i].shape[1]):
            self.assertArrayNear(np_read_head[i][j, k, :],
                                 tf_read_head[i][j, k, :],
                                 err=1e-8)

      # Verify that the NP and TF write heads have approximately the same
      # values.
      for i in range(len(np_write_head)):
        for j in range(np_write_head[i].shape[0]):
          for k in range(np_write_head[i].shape[1]):
            self.assertArrayNear(np_write_head[i][j, k, :],
                                 tf_write_head[i][j, k, :],
                                 err=1e-8)


  def testShapes(self):
    '''
    Verify that the correct state size is returned.
    '''

    with self.test_session() as session:
      #mem_size = (self.N, self.M)
      (num_slots, num_bits) = self.mem_size
      tf_regression_ntm = NTMRegression(self.mem_size, session,
                                        shift_range=self.shift_range)
      self.assertEqual(tf_regression_ntm.ntm_cell.state_size,
                       num_slots*(num_bits,) + 2*(num_slots,))


  def testOps(self):
    '''
    Verify that each of the operations (convolution, gating, etc.) are
    correct.
    Only compare the output from a single batch element and single time
    slice.
    '''

    mem_size = (self.N, self.M)
    initial_memory = self.initial_state[0:-2]
    np_initial_read_address = self.initial_state[-2]
    np_initial_write_address = self.initial_state[-1]
    tf_mem_prev = tf.stack(initial_memory, axis=1)
    np_mem_prev = np.stack(initial_memory, axis=1)
    # Only want the first batch element and first time slice from the
    # controller output to produce the read and write head values from a
    # single timestep.
    np_read_head, np_write_head = head_pieces(self.controller_output[0, 0, :],
                                              mem_size, self.S)

    np_read_ops_out = generate_address(np_read_head,
                                       np_initial_read_address[0, :],
                                       np_mem_prev[0, :, :],
                                       self.N,
                                       self.S)
    np_write_ops_out = generate_address(np_write_head[0:-2],
                                        np_initial_write_address[0, :],
                                        np_mem_prev[0, :, :],
                                        self.N,
                                        self.S)

    with self.test_session() as session:
      # The TF head pieces method takes in a single time slice from an entire
      # batch of controller data and spits out the read/write head values for
      # all batch items at that time slice.
      tf_write_head, tf_read_head = \
        NTMCell.head_pieces(self.controller_output[:, 0, :], mem_size, self.S)
      tf_read_ops_out = address_regression(tf_read_head,
                                           self.initial_state[-2],
                                           tf_mem_prev,
                                           self.N,
                                           self.S)
      tf_write_ops_out = address_regression(tf_write_head[0:-2],
                                            self.initial_state[-1],
                                            tf_mem_prev,
                                            self.N,
                                            self.S)

      tf_write_ops_out = session.run(tf_write_ops_out)
      tf_read_ops_out = session.run(tf_read_ops_out)

      self.assertEqual(len(tf_read_ops_out), len(np_read_ops_out))
      self.assertEqual(len(tf_write_ops_out), len(np_write_ops_out))

      for i in range(len(np_read_ops_out)):
        self.assertArrayNear(tf_read_ops_out[i][0], np_read_ops_out[i],
                             err=1e-8)
        self.assertArrayNear(tf_write_ops_out[i][0], np_write_ops_out[i],
                             err=1e-8)


  def testZeroInitialStateAndZeroControllerInput(self):
    '''
    Show that passing zeroes for the controller input and the initial state
    don't cause nan's or inf's on the output.
    '''

    zero_state = [np.zeros_like(i) for i in self.initial_state]
    zero_controller = np.zeros_like(self.controller_output)
    with self.test_session() as session:
      mem_size = (self.N, self.M)
      tf_regression_ntm = NTMRegression(mem_size, session, shift_range=self.S)

      output = tf_regression_ntm.run(zero_controller, zero_state)

      for item in output:
        #print(item[0])
        self.assertNotIn(np.inf, item)
        self.assertNotIn(-np.inf, item)
        self.assertNotIn(np.nan, item)


if __name__ == '__main__':
  test.main()
