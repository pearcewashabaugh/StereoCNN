import numpy as np
import matplotlib.pyplot as plt
import stereo_dist_sql as stdist


pic_height_pix = 200
subim1_size = 10
subim2_pad = 20

# iphone specific quantities:
# focal length (in m)
f = .035
# the size of an image rectangle in space
spat_height = .0289
spat_width = .0434

# Call the l2 difference calculator to find distance
leftim_n, rightim_n = stdist.image_preprocessor(pic_height_pix,
                                                'Data/Pictures_Data/left.jpeg',
                                                'Data/Pictures_Data/right.jpeg')
layer_0 = stdist.ImSplitDecisionLayer(leftim_n, rightim_n, subim1_size, subim2_pad)

# the master distance function giving the distance of each
# pixel from the origin (focal point of horizontal image)
def dist_function(i, j, ph, pw):
    return (ph-i)/100.0 + 1

# The physical coordinates of each pixel in 3D space in the horizontal image plane
spatial_horiz_coord = np.zeros((pic_height_pix, layer_0.pic_dim[1], 2))
for i in range(pic_height_pix):
    for j in range(layer_0.pic_dim[1]):
        spatial_horiz_coord[i, j] = [(pic_height_pix-i)*spat_height/pic_height_pix,
                                     j*spat_width/layer_0.pic_dim[1]]

# The physical coordinates of each pixel in 3D space with depth added
threed_spat_coord = np.zeros((pic_height_pix, layer_0.pic_dim[1], 3))
for i in range(pic_height_pix):
    for j in range(layer_0.pic_dim[1]):
        d = dist_function(i, j, pic_height_pix, layer_0.pic_dim[1])
        xh = spatial_horiz_coord[i, j][1]
        zh = spatial_horiz_coord[i, j][0]
        x = d*xh/(np.sqrt(f**2 + xh**2 + zh**2))
        z = d*zh/(np.sqrt(f**2 + xh**2 + zh**2))
        threed_spat_coord[i, j] = [x, np.sqrt(abs(d**2 - x**2 - z**2)), z]

# The physical coordinates of each pixel in 3D space in the vertical image plane
spatial_vert_coord = np.zeros((pic_height_pix, layer_0.pic_dim[1], 2))
vert_dist = 3
for i in range(pic_height_pix):
    for j in range(layer_0.pic_dim[1]):
        x = threed_spat_coord[i, j, 0]
        y = threed_spat_coord[i, j, 1]
        z = threed_spat_coord[i, j, 2]
        spatial_vert_coord[i, j] = [x*f/(vert_dist+f-z), y*f/(vert_dist+f-z)]
final_proj = np.zeros((pic_height_pix, layer_0.pic_dim[1], 3))

for i in range(pic_height_pix):
    for j in range(layer_0.pic_dim[1]):
        ip = pic_height_pix - int(spatial_vert_coord[i, j, 0] * pic_height_pix /
                                  spat_height)
        jp = int((spatial_vert_coord[i, j, 1]/spat_width)*layer_0.pic_dim[1])
        if (0 <= ip < pic_height_pix) & (0 <= jp < layer_0.pic_dim[1]):
            final_proj[ip, jp] = leftim_n[i, j]


plt.figure()
plt.imshow(final_proj)
plt.show()
# def vert_project(side_image, vert_dist, ph, pw):
#     proj_out = np.zeros((ph, pw))
#     for i, j in zip(range(ph), range(pw)):
#         x_prime = int(np.floor(dist_function(i, j, ph, pw)/(vert_dist - j)))
#         y_prime = int(np.floor(i/(vert_dist - j)))
#         proj_out[i, j] += side_image(x_prime, y_prime)
#     return proj_out
#
# final_proj = vert_project(leftim_n, 10, pic_height_pix, layer_0.pic_dim[1])
