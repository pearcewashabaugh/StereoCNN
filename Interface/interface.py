import sqlite3
import sys
# Stuff to be turned into command line arguments:

###############################################################################
# Picture chopping data

# Enter locations of left and right images:
lIm_loc = '../../../Projects_Data/StereoCNN_Data/Pictures_Data/left.jpeg'
rIm_loc = '../../../Projects_Data/StereoCNN_Data/Pictures_Data/right.jpeg'

# Specify desired resolution (higher resolutions take much longer):
pic_h_px = 100
pic_w_px = 133

# Will we be training this neural net or use a previously trained model?
TRAINING = True

# # If yes, specify the training data set location:
# train_data_loc = "Data/middlebury-perfect"

# The size of a subimage for training
s_lIm_size = 10

# The s_rIm_pad + s_lIm_size = s_rIm_w (the rightsubim width)
s_rIm_pad = 20

# The number of samples to take along the widthof each s_rIm. Must divide
pred_samp_w = 4

###############################################################################
# Camera characteristics

# focal length (m):
f_len = .0030

# camera baseline distance (m):
base_dist = .7
###############################################################################
# Neural net bulk hyperparameters

epochs = 2

batch_size = 100

# for train/test split
test_size = .1

random_state = 42

# for stochastic gradient descent
learning_rate = .05

decay_rate = 0

momentum = .3

###############################################################################
# Neural net layer hyperparameters

###############################################################################
# Save current job settings to data base

######
# Save picture chopping parameters
conn = sqlite3.connect('CNN/hyperparameters.db')
c = conn.cursor()
    
c.execute(
	'CREATE TABLE IF NOT EXISTS interface_params(job_id INT, pic_h_px INT, pic_w_px INT, s_lIm_size INT, s_rIm_pad INT, pred_samp_w INT)')

c.execute('SELECT COUNT(*) FROM interface_params')
is_empty = c.fetchall()[0][0]

if is_empty == 0:
	job_id = 0
	print("Hyperparameter database initialized.")
else:
	c.execute(
		'SELECT MAX(job_id) FROM interface_params'
		)
	job_id = c.fetchall()[0][0]
	job_id += 1

print('job_id =', job_id)

c.execute("INSERT INTO interface_params (job_id, pic_h_px, pic_w_px, s_lIm_size, s_rIm_pad, pred_samp_w) VALUES (?,?,?,?,?,?)", 
	(job_id, pic_h_px, pic_w_px, s_lIm_size, s_rIm_pad, pred_samp_w))
######
# Save neural net hyperparameters to db

c.execute(
	'CREATE TABLE IF NOT EXISTS neural_params(job_id INT, epochs INT, batch_size INT, test_size REAL, random_state INT, learning_rate REAL, decay_rate REAL, momentum REAL, json_model TEXT)')

c.execute("INSERT INTO neural_params (job_id, epochs, batch_size, test_size, random_state, learning_rate, decay_rate, momentum) VALUES (?,?,?,?,?,?,?,?)", (job_id, epochs, batch_size, test_size, random_state, learning_rate, decay_rate, momentum))

conn.commit()
conn.close()
###############################################################################