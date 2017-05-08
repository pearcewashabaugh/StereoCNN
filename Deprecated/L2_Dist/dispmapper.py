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

from conv_ply_stereo.image_preprocessor import image_preprocessor

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

pic_height_pix = 1000

leftim_n, rightim_n = image_preprocessor('Data/Pictures_Data/left.jpeg',
                                         'Data/Pictures_Data/right.jpeg',
                                         pic_height_pix)

conn = sqlite3.connect('L2_Dist/image_quantities.db')
c = conn.cursor()

c.execute("SELECT data FROM offset_map")
disp_quant = c.fetchall()

c.execute("SELECT shape0 FROM offset_map")
shape0 = c.fetchall()
shape0 = shape0[0][0]

c.execute("SELECT shape1 FROM offset_map")
shape1 = c.fetchall()
shape1 = shape1[0][0]

disp_quant = np.fromstring(disp_quant[0][0], dtype = float)

disp_quant = disp_quant.reshape((shape0,shape1))

shape0 = float(shape0)
shape1 = float(shape1)

x = np.arange(0,1,1/shape0)
y = np.arange(0,1,1/shape1)

interp_grid_x = np.arange(0,1,1/leftim_n.shape[0])
interp_grid_y = np.arange(0,1,1/leftim_n.shape[1])

dispmaptot = interpolate.interp2d(y,x,disp_quant, kind = 'cubic')

dispnew = dispmaptot(interp_grid_y,interp_grid_x)

dispnew += np.abs(np.min(dispnew))

dispnew = dispnew / np.max(dispnew)
dispnew += .5 * np.ones((dispnew.shape[0],dispnew.shape[1]))
dispnew = dispnew / np.max(dispnew)
dispnew = np.float32(dispnew)
#dispnew = np.float32(np.ones((dispnew.shape[0],dispnew.shape[1])))

h, w = leftim_n.shape[:2]
#f = 0.8*w                          # guess for focal length
f = 2*w
Q = np.float32([[1, 0, 0, -0.5*w],
                [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                [0, 0, 0,     -f], # so that y-axis looks up
                [0, 0, 1,      0]])
points = cv2.reprojectImageTo3D(dispnew, Q)
colors = cv2.cvtColor(np.float32(leftim_n), cv2.COLOR_BGR2RGB)
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

