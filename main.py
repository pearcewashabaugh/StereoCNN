
###############################################################################
# Import user input. See file for parameters.

import Interface.interface as uinp

###############################################################################
# Train

if uinp.TRAINING == True:
	import CNN.train 

CNN.train.trainer(uinp.pic_h_px, uinp.pic_w_px, uinp.s_lIm_size, 
	uinp.s_rIm_pad)
###############################################################################
# Pre-Process input images

from Image_PrePost_Processing.image_preprocessor import image_resizer, subim_maker

lIm = image_resizer(uinp.lIm_loc,
                        uinp.pic_h_px,
                        pic_width_pix = uinp.pic_w_px)
rIm = image_resizer(uinp.rIm_loc,
                        uinp.pic_h_px,
                        pic_width_pix = uinp.pic_w_px)

###############################################################################
# Evaluate

from CNN.eval import evaluator

evaluator(uinp.s_lIm_size, uinp.s_rIm_pad, uinp.pred_samp_w, lIm, rIm)

###############################################################################
# Output 3D .ply file

import Image_PrePost_Processing.dispmapper as dispmap

dispmap.projector3D(uinp.pic_h_px, lIm, rIm)



