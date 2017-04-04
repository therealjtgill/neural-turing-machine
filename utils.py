
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

def save_output_plot(targets, predictions, folder, filename):

	path = make_dir(folder)
	
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

	filename = str(filename) + '.png'
	
	plt.savefig(os.path.join(path, filename))
	plt.close()

def save_address_plot(addresses, folder, filename):

	path = make_dir(folder)
	#print(addresses)
	plt.imshow(addresses.T, interpolation='none', cmap='gray')
	plt.xlabel('time')
	plt.ylabel('address')

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

def save_error(error, folder, filename):

	path = make_dir(folder)
	filename = str(filename) + '.err'

	with open(os.path.join(path, filename), 'a') as f:
		f.write(str(error) + '\n')