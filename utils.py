
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

def make_dir(folder):
    cwd = os.getcwd()
    save_dir = os.path.join(cwd, folder)

    if os.path.isdir(save_dir):
        return save_dir
    else:
        os.mkdir(save_dir)
        return save_dir

def save_double_plot(plot1, plot2, folder, filename, ylabel1='', ylabel2=''):

    path = make_dir(folder)
    
    if len(plot1.shape) > 2:
        plot1 = plot1[0]

    if len(plot2.shape) > 2:
        plot2 = plot2[0]

    plot1 = plot1.T
    plot2 = plot2.T

    #print(plot1)
    #print(plot2)

    plot1_extent = [0, plot1.shape[1], 0, plot1.shape[0]]
    plot2_extent = [0, plot2.shape[1], 0, plot2.shape[0]]

    fig = plt.gcf()
    fig.set_size_inches(16, 9)
    plt.figure(1)
    ax1 = plt.subplot(211)
    ax1.imshow(plot1, interpolation='none', extent=plot1_extent)
    #ax1.axis('off')
    #plt.title('plot1')
    ax1.set_ylabel(ylabel1)
    
    ax2 = plt.subplot(212, sharex=ax1)
    ax2.imshow(plot2, interpolation='none', extent=plot2_extent)
    #ax2.axis('off')
    #plt.title('plot2')
    ax2.set_ylabel(ylabel2)

    filename = str(filename) + '.png'
    
    plt.savefig(os.path.join(path, filename), dpi=100)
    plt.close()

def save_single_plot(val, folder, filename, ylabel=''):

    path = make_dir(folder)
    #print(addresses)
    plt.imshow(val.T, interpolation='none', cmap='gray', vmin=0., vmax=1.)
    plt.xlabel('time')
    plt.ylabel(ylabel)

    filename = str(filename) + '.png'

    plt.savefig(os.path.join(path, filename))
    plt.close()

def save_multi_plot(plots, folder, filename, ylabels=None):

    path = make_dir(folder)
    n = len(plots)

    if ylabels == None:
        ylabels = ('',)*n

    figure = plt.figure()
    ax = []
    for i in range(n):
        if i > 0:
            ax.append(figure.add_subplot(n*100 + 10 + i+1, sharex=ax[0]))
        else:
            ax.append(figure.add_subplot(n*100 + 10 + i+1))

        ax[i].matshow(plots[i].T, interpolation='none',
            cmap='gray', vmin=0., vmax=1., aspect='auto')
        plt.setp(ax[i].get_xticklabels(), fontsize=6)
        plt.xlabel('time')
        plt.ylabel(ylabels[i])

    filename = str(filename) + '.png'

    plt.savefig(os.path.join(path, filename))
    plt.close()

def get_training_batch(batch_size, seq_length, num_bits):

    bs = batch_size
    sl = seq_length
    nb = num_bits
    batch_x = np.zeros((bs, sl*2+1, nb+1))
    batch_y = np.zeros((bs, sl*2+1, nb))
    sequence = (np.random.rand(bs, sl, nb)*2).astype(int)
    
    batch_x[:,0:sl,0:nb] = sequence[:,:,:]
    #batch_y[:,0:sl,0:nb] = sequence[:,:,:]
    batch_y[:,sl+1:2*sl+1,0:nb] = sequence[:,:,:]
    batch_x[:,sl,num_bits] = 1
    #batch_x = batch_y[:,:,:]
    #batch_y[:,sl+1:,0:nb] = sequence[:,:,:]
    #batch_y = batch_y[:,:,0:num_bits]
    batch_x[:,sl+1:sl*2+1,:] = 0

    #print(str(batch_x[0,0:2,:]))

    return batch_x, batch_y

def save_text(data, folder, filename):

    path = make_dir(folder)
    filename = str(filename) + '.err'

    with open(os.path.join(path, filename), 'a') as f:
        f.write(str(data) + '\n')