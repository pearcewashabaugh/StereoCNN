#Code partially pilfered from 
#https://github.com/opencv/opencv/blob/master/samples/python/stereo_match.py

import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from skimage.transform import resize
from skimage import io
import sqlite3
import sys
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D
import cv2
from scipy.ndimage.filters import gaussian_filter

from image_preprocessor import image_resize

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    colors = colors*255.0
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

pic_height_pix = 100

lIm, disp = image_resize('../Data/Adirondack-perfect/im0.png',
                        '../Data/Adirondack-perfect/disp0_000.png',
                        pic_height_pix)
disp = disp[:,:,0]
disp = np.float32(disp)

# x = np.arange(0,1,1/shape0)
# y = np.arange(0,1,1/shape1)

# interp_grid_x = np.arange(0,1,1/leftim_n.shape[0])
# interp_grid_y = np.arange(0,1,1/leftim_n.shape[1])

# dispmaptot = interpolate.interp2d(y,x,disp_quant, kind = 'cubic')

# dispnew = dispmaptot(interp_grid_y,interp_grid_x)
disp += np.abs(np.min(disp))

disp = disp / np.max(disp)
disp += np.ones((disp.shape[0],disp.shape[1]))
disp = disp / np.max(disp)

#disp = gaussian_filter(disp, sigma=4)
# l = 15.0
# li = 15
# for i in range(int(disp.shape[0]/l)):
#     for j in range(int(disp.shape[1]/l)):
#         m = np.max(disp[li*i:(li*i+li),li*j:(li*j+li)])
#         disp[li*i:(li*i+li),li*j:(li*j+li)] = m*np.ones((li,li))

disp = np.float32(disp)
#dispnew = np.float32(np.ones((dispnew.shape[0],dispnew.shape[1])))

# dispnew = np.float32(np.ones((dispnew.shape[0],dispnew.shape[1])))

h, w = lIm.shape[:2]
f = w                          # guess for focal length
#f = 2*w
Q = np.float32([[1, 0, 0, -0.5*w],
                [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                [0, 0, 0,     -f], # so that y-axis looks up
                [0, 0, 1,      0]])
points = cv2.reprojectImageTo3D(disp, Q)
colors = cv2.cvtColor(np.float32(lIm), cv2.COLOR_BGR2RGB)
# print(np.max(dispnew))
# print(np.min(dispnew))

out_fn = 'out.ply'
#write_ply('out.ply', out_points, out_colors)
write_ply('out.ply', points, colors)
print('%s saved' % 'out.ply')
# plt.figure()
# plt.title('Displacement Distance')
# plt.imshow(colors)

# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Make data
# # u = np.linspace(0, 2 * np.pi, 100)
# # v = np.linspace(0, np.pi, 100)
# # x = 10 * np.outer(np.cos(u), np.sin(v))
# # y = 10 * np.outer(np.sin(u), np.sin(v))
# # z = 10 * np.outer(np.ones(np.size(u)), np.cos(v))

# # Plot the surface
# ax.scatter(interp_grid_y, interp_grid_x, dispnew)

# plt.show()
# plt.figure()
# plt.title('Displacement Distance')
# plt.imshow(disp_quant, cmap='hot', interpolation='nearest')

#plt.show()

