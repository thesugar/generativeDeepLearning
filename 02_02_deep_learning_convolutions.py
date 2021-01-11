# %%
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from scipy.ndimage import correlate
import numpy as np
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import rescale, resize

# %%
im = rgb2gray(data.coffee())
im = resize(im, (64, 64))
print(im.shape)

plt.axis('off')
plt.imshow(im, cmap= 'gray')
# %%
# horizontal edge filter
filter1 = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1, -1, -1],
])

new_image = np.zeros(im.shape)

# e.g.) np.pad(np.array([1,2,3]), 1, 'constant')
#     --> array([0, 1, 2, 3, 0])
im_pad = np.pad(im, 1, 'constant')

for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        try:
            new_image[i,j] = im_pad[i-1,j-1] * filter1[0,0] + im_pad[i-1,j] * filter1[0,1] + im_pad[i-1,j+1] * filter1[0,2] + \
                             im_pad[i,j-1] * filter1[1,0] + im_pad[i,j] * filter1[1,1] + im_pad[i,j+1] * filter1[1,2] + \
                             im_pad[i+1,j-1] * filter1[2,0] + im_pad[i+1,j] * filter1[2,1] + im_pad[i+1,j+1] * filter1[2,2]
        except:
            pass

plt.axis('off')
plt.imshow(new_image, cmap='Greys')
# %%
# vertical edge filter

filter2 = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1],
])

new_image_2 = np.zeros(im.shape)

im_pad_2 = np.pad(im, 1, 'constant')

for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        try:
            new_image_2[i,j] = im_pad_2[i-1,j-1] * filter2[0,0] + im_pad_2[i-1,j] * filter2[0,1] + im_pad_2[i-1,j+1] * filter2[0,2] + im_pad_2[i,j-1] * filter2[1,0] + im_pad_2[i,j] * filter2[1,1] + im_pad_2[i,j+1] * filter2[1,2] + im_pad_2[i+1,j-1] * filter2[2,0] + im_pad_2[i+1,j] * filter2[2,1] + im_pad_2[i+1,j+1] * filter2[2,2]
        except:
            pass

plt.axis('off')
plt.imshow(new_image_2, cmap='Greys')
# %%
# horizontal edge filter with stride 2

stride = 2

new_image_3 = np.zeros((int(im.shape[0] / stride), int(im.shape[1] / stride)))
print(f'the shape of new_image_3 is {new_image_3.shape}')

im_pad_3 = np.pad(im, 1, 'constant')

for i in range(0, im.shape[0], stride):
    for j in range(0, im.shape[1], stride):
        try:
            new_image_3[int(i/stride), int(j/stride)] = im_pad_3[i-1,j-1] * filter1[0,0] + im_pad_3[i-1,j] * filter1[0,1] + im_pad_3[i-1,j+1] * filter1[0,2] + im_pad_3[i,j-1] * filter1[1,0] + im_pad_3[i,j] * filter1[1,1] + im_pad_3[i,j+1] * filter1[1,2] + im_pad_3[i+1,j-1] * filter1[2,0] + im_pad_3[i+1,j] * filter1[2,1] + im_pad_3[i+1,j+1] * filter1[2,2]
        except:
            pass

plt.axis('off')
plt.imshow(new_image_3, cmap='Greys')
# %%
# vertical edge filter with stride 2

new_image_4 = np.zeros((int(im.shape[0] / stride), int(im.shape[0] / stride)))

im_pad_4 = np.pad(im, 1, 'constant')

for i in range(0, im.shape[0], stride):
    for j in range(0, im.shape[1], stride):
        try:
            new_image_4[int(i/stride),int(j/stride)] = im_pad_4[i-1,j-1] * filter2[0,0] + im_pad_4[i-1,j] * filter2[0,1] + im_pad_4[i-1,j+1] * filter2[0,2] +\
                                                     im_pad_4[i,j-1] * filter2[1,0] + im_pad_4[i,j] * filter2[1,1] + im_pad_4[i,j+1] * filter2[1,2] + \
                                                     im_pad_4[i+1,j-1] * filter2[2,0] +im_pad_4[i+1,j] * filter2[2,1] +im_pad_4[i+1,j+1] * filter2[2,2] 
        except:
            pass

plt.axis('off')
plt.imshow(new_image_4, cmap='Greys')
