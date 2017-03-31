import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import warnings

warnings.filterwarnings('ignore')

def save_matrix_plots(targets, predictions, filename, offset=0):

	if len(targets.shape) > 2:
		targets = targets[0]

	if len(predictions.shape) > 2:
		predictions = predictions[0]

	targets = targets.T
	predictions = predictions.T

	targets_extent = [0, targets.shape[1], 0, targets.shape[0]]
	predictions_extent = [0, predictions.shape[1], 0, predictions.shape[0]]

	plt.figure(1)
	ax1 = plt.subplot(211)
	ax1.imshow(targets, interpolation='none', extent=targets_extent)
	#ax1.axis('off')
	#plt.title('Targets')
	ax1.set_ylabel('Targets')
	

	ax2 = plt.subplot(212, sharex=ax1)
	ax2.imshow(predictions, interpolation='none', extent=predictions_extent)
	#ax2.axis('off')
	#plt.title('Predictions')
	ax2.set_ylabel('Predictions')
	
	plt.savefig(str(filename) + str(offset) + '.png')

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

def save_error(error, filename):
	with open(filename, 'a') as f:
		f.write(str(error) + '\n')