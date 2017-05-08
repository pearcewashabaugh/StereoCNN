
###############################################################################
# Stuff to be turned into command line arguments:

# Enter locations of left and right images:
lIm_loc = '../Data/Pictures_Data/left.jpeg'
rIm_loc = '../Data/Pictures_Data/right.jpeg'

# Specify desired resolution (higher resolutions take much longer):
pic_h_px = 500
pic_w_px = 667

# Will we be training this neural net or use a previously trained model?
TRAINING = True

# # If yes, specify the training data set location:
# train_data_loc = "Data/middlebury-perfect"

# Please input camera characteristics:

# focal length (m):
f_len = .0030

# camera baseline distance (m):
base_dist = .7

# Enable advanced inputs to tune results?
ADV_INP = True

if ADV_INP == True:
	# The size of a subimage for training
	s_lIm_size = 10

	# The s_rIm_pad + s_lIm_size = s_rIm_w (the right subim width)
	s_rIm_pad = 10

	# The number of samples to take along the width of each s_rIm. Must divide
	# s_rIm_pad
	pred_samp_w = 5

else:
	# The size of a subimage for training
	s_lIm_size = 10

	# The s_rIm_pad + s_lIm_size = s_rIm_w (the right subim width)
	s_rIm_pad = 5

	# The number of samples to take along the width of each s_rIm. Must divide
	# s_rIm_pad
	pred_samp_w = 1
###############################################################################
