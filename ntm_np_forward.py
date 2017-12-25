import numpy as np
from ntm_cell import NTMCell

DEBUG = False

def softmax(vec):
    return np.exp(vec)/np.sum(np.exp(vec), axis=-1, keepdims=True)


def sigmoid(vec):
    return 1./(1. + np.exp(-1*vec))


def relu(vec):
    return np.array([0. if v < 0. else v for v in vec])


def softplus(vec):
    # softplus(x) = log(1 + e^x)
    return np.log(1. + np.exp(vec))


def head_pieces(head, mem_size, shift_range):
    # Assumes controller output has the shape (batch_size, 4*M + 2*S + 6).
    # Returns tuples containing batches of the head pieces
    (N, M) = mem_size
    S = shift_range
    
    center = int(S/2)
    shift_bias = np.zeros(S)
    #shift_bias[center + 1] = 2.5
    
    first_split = [M + S + 3]
    read_head_raw, write_head_raw = np.split(head, first_split, axis=-1)
    
    write_pieces = np.split(write_head_raw, [M, M+S, M+S+1, M+S+2, M+S+3, 2*M+S+3,], axis=-1)
    
    read_pieces = np.split(read_head_raw, [M, M+S, M+S+1, M+S+2], axis=-1)
    
    key_w = write_pieces[0]
    shift_w = softmax(write_pieces[1] + shift_bias)
    gamma_w = softplus(write_pieces[2]) + 1.
    beta_w = softplus(write_pieces[3])
    g_w = sigmoid(write_pieces[4])
    add_w = sigmoid(write_pieces[5])
    erase_w = sigmoid(write_pieces[6])
    
    key_r = read_pieces[0]
    shift_r = softmax(read_pieces[1] + shift_bias)
    gamma_r = softplus(read_pieces[2]) + 1.
    beta_r = softplus(read_pieces[3])
    g_r = sigmoid(read_pieces[4])
    
    write_head = (key_w, shift_w, gamma_w, beta_w, g_w, add_w, erase_w)
    read_head = (key_r, shift_r, gamma_r, beta_r, g_r)
    
    return read_head, write_head


def content_addressing(mem, k, B=1.0):
    # Compute the cosine similarity between 'k' and every
    # row of 'mat', then multiply the results by 'B', and
    # perform a softmax operation.
    # k.shape = [M,]
    # mat.shape = [N, M]
    # B = float
    if DEBUG:
        print('Content addressing:')
        print('\t', k.shape, 'Should be (1, M)')
        print('\t', mat.shape, 'Should be (N, M)')
    
    #numerator = np.dot(mat, k.T)
    numerator = np.dot(mem, np.expand_dims(k, axis=1))
    denominator = np.sqrt(np.diag(np.dot(mem, mem.T)))*np.sqrt(np.dot(k, k.T))
    similarities = np.diag(numerator/denominator)
    if DEBUG:
        print('\tSimilarities shape:', similarities.shape)
        print('\tOutput:', softmax(B*similarities))
    return softmax(B*similarities)


def interpolation(g, w_prev, w_content):
    # Interpolates between the previously used address and
    # the address generated from content.
    if DEBUG:
        print('Interpolation:')
        print('gate:', g)
        print('\tOutput:', g*w_content + (1. - g)*w_prev)
    return g*w_content + (1. - g)*w_prev


def circular_convolution(a, b):
    if DEBUG:
        print('Circular convolution:')
        print('\t', a.shape)
        print('\t', b.shape)
    conv = np.zeros(len(a))
    for i in range(len(a)):
        for j in range(len(a)):
            #print((i - j) % len(b))
            conv[i] += a[j]*b[(i - j) % len(b)]
    if DEBUG:
        print('\tOutput:', conv)
    return conv


def conv_shift(w, s):
    S = len(s)
    center = int(S/2)
    s = np.pad(s, (0, np.abs(len(w) - len(s))), 'constant')
    #print(s, len(s))
    center_split = np.split(s, [center])
    #print(len(center_split[0]), len(center_split[1]))
    s = np.concatenate([center_split[1], center_split[0]], axis=0)
    
    if DEBUG:
        print('Shift:')
        print('\t', s, len(s))
    
    return circular_convolution(w, s)


def sharpen(w, y):
    # Sharpens the array 'w' by raising each element to the
    # power of 'y' and normalizing.
    if DEBUG:
        print('Sharpen:')
        print('\tw: ', w)
        print('\ty: ', y)
        print('\tOutput:', np.power(w, y)/np.sum(np.power(w, y)))

    return np.power(w, y)/np.sum(np.power(w, y))


def write_memory(w, mat, erase, add):
    # erase.shape = (M, 1)
    # add.shape = (M, 1)
    # mat.shape = (N, M)
    # w.shape = (N, 1)
    # Note that several of these ops are outer products, and the
    # first multiplication is element-wise multiplication, not 
    # matrix multiplication.
    
    w = np.expand_dims(w, axis=1)
    erase = np.expand_dims(erase, axis=1)
    add = np.expand_dims(add, axis=1)
    
    if DEBUG:
        print('Write Memory')
        print('\t', w.shape)
        print('\t', mat.shape)
        print('\t', erase.shape)
        print('\t', add.shape)

    return mat*(1.-np.dot(w, erase.T)) + np.dot(w, add.T)


def read_memory(w, mat):
    # Retrieves a portion of the memory matrix according to the 
    # address supplied in 'w'.
    return np.dot(w.T, mat)


# ### Testing Strategy
# 
# The ntm_cell is recurrent, so we need to verify that the memory and attention are handled correctly in successive timesteps across a batch of sequences.
# 
# #### Numpy
# 
# 1. Generate a batch of fake outputs from the controller **[batch_size, sequence_length, 4\*M + 2\*S + 6]**; save these
# 2. Use an initial state generated by the ntm_cell class to kick off the process; save this
# 3. Break batches into row vectors, compute batch output at each timestep
# 4. Save all timestep output
# 
# #### TensorFlow
# 
# 1. Use same fake controller outputs as Numpy
# 2. Use same initial memory/attention state as Numpy
# 3. Calculate batch output for each tiemstep, save it
# 4. Save all timestep output
# 
# 
# * Compute L2 norm of difference between batch outputs at each timestep
# * Average the L2 difference between timesteps, this is the error
# * If the error is below some threshold, build passes


def generate_address(pieces, w_prev, mem_prev, N, S):
    
    key, shift, gamma, beta, g = pieces
    
    w_c = content_addressing(mem_prev, key, B=beta)
    #w_i = g*w_c + (1. - g)*w_prev
    w_i = interpolation(g, w_prev, w_c)
    #w_conv = circular_convolution(w_i, shift)
    #w_sharp = sharpen(w_conv, gamma)
    w_shift = conv_shift(w_i, shift)
    w_sharp = sharpen(w_shift, gamma)
    #w_sharp = np.power(w_conv, gamma)
    #w = w_sharp/np.sum(w_sharp)
    
    return w_c, w_i, w_shift, w_sharp


def numpy_forward_pass(N, M, shift_range, initial_state, controller_output,
    out_directory='.'):
    # The numpy forward pass only works on vectors, not matrices. So the
    # initial state has to be broken down into individual vectors.
    # THIS SHOULD ONLY LOOP OVER A SINGLE TIME SEQUENCE
    
    # controller_output -> (seq_length, 4*M + 2*S + 6)
    # shift_range       -> S
    # initial_state     -> (N*(M,) + 2*(N,))
    #   {N memory cells, 2 attention vectors}
    
    #print('Numpy forward pass')
    #print('Initial state shape:', initial_state.shape)
    #print('Controller output shape:', controller_output.shape)

    seq_length = controller_output.shape[0]
    mem_size = (N, M)
    S = shift_range
    
    read_addresses = np.zeros((seq_length, N))
    write_addresses = np.zeros((seq_length, N))
    reads = np.zeros((seq_length, M))
    #writes = np.zeros((seq_length, M))
    #print('Len: ', len(initial_state))
    mem_prev = np.stack(tuple(initial_state[0:-2]), axis=0)
    #print('Memory shape: ', mem_prev.shape)
    w_read_prev = initial_state[-2]
    w_write_prev = initial_state[-1]
    
    for i in range(seq_length):
        if DEBUG:
            print('----------------------')
            print('Sequence position ',  i)
            print('----------------------')
        read_head, write_head = head_pieces(controller_output[i, :], mem_size, S)
        #read_addresses[i, :] = generate_address(read_head, w_read_prev, mem_prev, N, S)
        #write_addresses[i, :] = generate_address(write_head[0:-2], w_write_prev, mem_prev, N, S)
        read_head_ops = generate_address(read_head, w_read_prev, mem_prev, N, S)
        write_head_ops = generate_address(write_head[0:-2], w_write_prev, mem_prev, N, S)

        read_addresses[i, :] = read_head_ops[-1]
        write_addresses[i, :] = write_head_ops[-1]
        
        w_read_prev = read_addresses[i, :].copy()
        w_write_prev = write_addresses[i, :].copy()
        
        add = write_head[-2]
        erase = write_head[-1]
        
        mem_current = write_memory(write_addresses[i, :], mem_prev, erase, add)
        reads[i, :] = read_memory(read_addresses[i, :], mem_current)
        
        mem_prev = mem_current.copy()

        if i == 0:
            open_as = 'w'
        else:
            open_as = 'a'
        with open('numpystuff.dat', open_as) as f:
            for i in range(4):
                f.write(str(read_head_ops[i]) + '\n')

    return read_addresses, write_addresses, reads

# ----------------------------------------------------------------------------#



