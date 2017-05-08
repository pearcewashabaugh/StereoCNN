#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.
Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
end_header
'''

# def write_ply(fn, verts, colors):
#     verts = verts.reshape(-1, 3)
#     colors = colors.reshape(-1, 3)
#     verts = np.hstack([verts, colors])
#     with open(fn, 'wb') as f:
#         f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
#         np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
def write_ply(fn, verts):
    verts = verts.reshape(-1, 3)
    verts = np.hstack([verts])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f')


if __name__ == '__main__':
    print('loading images...')
    imgL = cv2.pyrDown(cv2.pyrDown(cv2.imread('Data/Pictures_Data/left.jpeg',0)))
    imgR = cv2.pyrDown(cv2.pyrDown(cv2.imread('Data/Pictures_Data/right.jpeg',0)))

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp

    print('computing disparity...')
    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
    disp = stereo.compute(imgL,imgR).astype(np.float32) / 5.0

    print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    f = 0.8*w                          # guess for focal length
    #f = 0.001*w
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv2.reprojectImageTo3D(disp, Q, handleMissingValues=True)
    #colors = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)

    #mask = disp > disp.min()
    #out_points = points[mask]
    #out_colors = colors[mask]
    out_fn = 'out.ply'
    #write_ply('out.ply', out_points, out_colors)
    write_ply('out.ply', points)
    print('%s saved' % 'out.ply')

    cv2.imshow('left', imgL)
    cv2.imshow('disparity', (disp)/num_disp)
    cv2.waitKey()
    cv2.destroyAllWindows()