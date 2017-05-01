import tensorflow as tf
import numpy as np
from ntm import *

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

def associative_recall_task_batch(batch_size, seq_length,
    num_sequences, num_bits=6):
    
    bs = batch_size
    sl = seq_length
    ns = num_sequences
    nb = num_bits

    #batch_x = np.zeros((bs, (sl*ns+ns)*2 + 1, nb+2))
    #batch_y = np.zeros((bs, (sl*ns+ns)*2 + 1, nb+2))

    sequences = []
    for _ in range(ns):
        sequences.append(np.random.randint(2, size=[bs, sl, nb]) + 0.)

    # Add space at the bottom of the training sequence so that the 
    # delimiters will fit.
    buf = np.zeros((bs, sl, 2))
    for i, s in enumerate(sequences):
        sequences[i] = np.append(s, buf, axis=2)

    pre_x = []
    store_delim = np.zeros((bs, 1, nb+2))
    recall_delim = np.zeros((bs, 1, nb+2))
    store_delim[:, 0, nb] = 1.
    recall_delim[:, 0, nb+1] = 1.
    for i, s in enumerate(sequences):
        pre_x.append(s)
        pre_x.append(store_delim)

    recall_symbol_index = np.random.randint(len(sequences))
    pre_x.append(sequences[recall_symbol_index])
    pre_x.append(recall_delim)
    pre_x.append(np.zeros_like(sequences[0]) + 0.)

    batch_x = np.concatenate(tuple(pre_x), axis=1)
    pre_y = [np.zeros_like(bx) for bx in pre_x[0:-1]]
    pre_y.append(sequences[(recall_symbol_index + 1) % ns])
    batch_y = np.concatenate(tuple(pre_y), axis=1)[:,:,0:nb]


    #print('training batch shapes:')
    #print(batch_x.shape)
    #print(batch_y.shape)

    return batch_x, batch_y

def train_copy_task(ntm, date, save_dir, session):

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
                    print('sequence length:', i,
                        'average err:', sequence_err[i])
            sequence_err = [0.,]*20
            sequences = [0.,]*20
            save_text(error, save_dir, '0-train')
            avg_error = 0.
        else:
            sequence_err[seq_length] += error
            sequences[seq_length] += 1.
            
        if step % save_threshold == 0:
            saver.save(session, os.path.join(save_dir, "model.ntm.ckpt"))
            seq_length = (10, 20, 30, 50, 100)

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

def train_associative_recall_task(ntm, date, save_dir, session):
    
    batch_size = 64
    avg_error = 0
    lr = 1e-4
    input_size = 8
    output_size = 6
    max_batches = 50000
    print_threshold = 100
    save_threshold = 500

    saver = tf.train.Saver(tf.global_variables())

    for step in range(max_batches):
        #seq_length = 8 + int(np.random.rand()*12)
        seq_length = 3
        num_sequences = 2 + int(np.random.rand()*5)
        batch_in, batch_out = associative_recall_task_batch(batch_size,
            seq_length, num_sequences)

        error = ntm.train_batch(batch_in, batch_out)
        avg_error += error/print_threshold

        if step % print_threshold == 0:

            print('step: {0} average error: {1}'.format(step, avg_error))
            print('average sequence errors:')
            save_text(error, save_dir, '0-train')
            avg_error = 0.
            
        if step % save_threshold == 0:
            saver.save(session, os.path.join(save_dir, "ar_model.ntm.ckpt"))
            seq_length = (2, 3, 5, 8)

            for s in seq_length:
                test_x, test_y = associative_recall_task_batch(2, 
                    s, num_sequences)

                pred, w_r, w_w, g_r, g_w, s_r, s_w = ntm.run_once(test_x)

                suffix = str(s) + '-' + str(step)

                save_single_plot(test_x, save_dir, 'input' + suffix, 
                    'input')
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

def main():

    mem_shape=(32,15)
    input_size = 8
    output_size = 6
    session = tf.Session()
    ntm = NTM(mem_shape, input_size, output_size, session)
    session.run(tf.global_variables_initializer())
    print('Computation graph built.')

    saver = tf.train.Saver(tf.global_variables())

    date = datetime.now()
    date = str(date).replace(' ', '').replace(':', '-')
    save_dir = make_dir(date)

    #train_model(ntm, batch_size=64, max_batches=5e4, save_threshold=100)

    #train_copy_task(ntm, date, save_dir, saver, saver)
    train_associative_recall_task(ntm, date, save_dir, session)


if __name__ == '__main__':
    main()