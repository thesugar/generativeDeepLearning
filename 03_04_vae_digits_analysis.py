# %%
from IPython import get_ipython

get_ipython().run_line_magic('load_ext', 'autoreload') #type: ignore
get_ipython().run_line_magic('autoreload', '2') #type: ignore

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm  # type:ignore

from models.VAE import VariationalAutoencoder
from utils.loaders import load_mnist, load_model

# %%
# run params
SECTION = 'vae'
RUN_ID = '0002'
DATA_NAME = 'digits'
RUN_FOLDER = f'run/{SECTION}/'
RUN_FOLDER += '_'.join([RUN_ID, DATA_NAME])

# %%
(x_train, y_train), (x_test, y_test) = load_mnist()

# %%
vae = load_model(VariationalAutoencoder, RUN_FOLDER)

# ## reconstructing original paintings

n_to_show = 10
example_idx = np.random.choice(range(len(x_test)), n_to_show)
example_images = x_test[example_idx]

_, _, z_points = vae.encoder.predict(example_images) # predict の戻り値は mu（μ）, log_var（σ から求まる）, z (= μ + σ * ε)。VAE.py を参照。
                                                     # いずれも shape は (5000, 2)。この 2 というのは z_dim として 03_03 で指定している。

reconst_images = vae.decoder.predict(z_points)

fig = plt.figure(figsize=(15, 3))
fig.subplots_adjust(hspace=0.4, wspace=0.4)  # type: ignore

for i in range(n_to_show):
    img = example_images[i].squeeze()
    sub = fig.add_subplot(2, n_to_show, i+1)  # type: ignore
    sub.axis('off')
    sub.text(0.5, -0.35, str(np.round(z_points[i], 1)), fontsize=10, ha='center', transform=sub.transAxes)

    sub.imshow(img, cmap='gray_r')

for i in range(n_to_show):
    img = reconst_images[i].squeeze()
    sub = fig.add_subplot(2, n_to_show, i+n_to_show+1)  # type: ignore
    sub.axis('off')
    sub.imshow(img, cmap='gray_r')
# %%
# Mr N. Coder's Wall

n_to_show = 5000
figsize = 12

example_idx = np.random.choice(range(len(x_test)), n_to_show)
example_images = x_test[example_idx]
example_labels = y_test[example_idx]

_, _, z_points = vae.encoder.predict(example_images)

print('z_points is\n', z_points)

min_x = min(z_points[:, 0])
max_x = max(z_points[:, 0])
min_y = min(z_points[:, 1])
min_y = min(z_points[:, 1])

plt.figure(figsize=(figsize, figsize))
plt.scatter(z_points[:, 0] , z_points[:, 1], c='black', alpha=0.5, s=2)
plt.show()
# %%
# The new fenerated art exhibition
figsize = 8
plt.figure(figsize=(figsize, figsize))
plt.scatter(z_points[:, 0], z_points[:, 1], c='black', alpha=0.5, s=2)

grid_size = 15
grid_depth = 2
figsize = 15

x = np.random.normal(size=grid_size*grid_depth)
y = np.random.normal(size=grid_size*grid_depth)

z_grid = np.array(list(zip(x, y)))
reconst = vae.decoder.predict(z_grid)

plt.scatter(z_grid[:, 0], z_grid[:, 1], c = 'red', alpha=1, s=20)
plt.show()

fig = plt.figure(figsize=(figsize, grid_depth))
fig.subplots_adjust(hspace=0.4, wspace=0.4)  # type: ignore

for i in range(grid_size*grid_depth):
    ax = fig.add_subplot(grid_depth, grid_size, i+1)  # type: ignore
    ax.axis('off')
    ax.text(0.5, -0.35, str(np.round(z_grid[i],1)), fontsize=8, ha='center', transform=ax.transAxes)
    
    ax.imshow(reconst[i, :,:,0], cmap = 'Greys')
# %%
n_to_show = 5000
grid_size = 15
fig_height = 7
fig_width = 15

example_idx = np.random.choice(range(len(x_test)), n_to_show)
example_images = x_test[example_idx]
example_labels = y_test[example_idx]

_, _, z_points = vae.encoder.predict(example_images)
p_points = norm.cdf(z_points)  # cdf: Cumulative distribution function 累積分布関数 / 確率変数がある値以下となる確率

fig = plt.figure(figsize=(fig_width, fig_height))

ax = fig.add_subplot(1, 2, 1)  # type: ignore
plot_1 = ax.scatter(z_points[:, 0] , z_points[:, 1] , cmap='rainbow' , c= example_labels
            , alpha=0.5, s=2)
plt.colorbar(plot_1)

ax = fig.add_subplot(1, 2, 2)  # type: ignore
plot_2 = ax.scatter(p_points[:, 0] , p_points[:, 1] , cmap='rainbow' , c= example_labels
            , alpha=0.5, s=5)

plt.show()
# %%
n_to_show = 5000
grid_size = 20
figsize = 8

example_idx = np.random.choice(range(len(x_test)), n_to_show)
example_images = x_test[example_idx]
example_labels = y_test[example_idx]

_, _, z_points = vae.encoder.predict(example_images)

plt.figure(figsize=(5, 5))
plt.scatter(z_points[:, 0] , z_points[:, 1] , cmap='rainbow' , c= example_labels  # type: ignore
            , alpha=0.5, s=2)
plt.colorbar()

x = norm.ppf(np.linspace(0.01, 0.99, grid_size))  # ppf: percent point function（累積分布関数 cdf の逆関数）
y = norm.ppf(np.linspace(0.01, 0.99, grid_size))
xv, yv = np.meshgrid(x, y)
xv = xv.flatten()
yv = yv.flatten()
z_grid = np.array(list(zip(xv, yv)))

reconst = vae.decoder.predict(z_grid)

plt.scatter(z_grid[:, 0] , z_grid[:, 1], c = 'black'#, cmap='rainbow' , c= example_labels
            , alpha=1, s=2)

plt.show()


fig = plt.figure(figsize=(figsize, figsize))
fig.subplots_adjust(hspace=0.4, wspace=0.4)  # type: ignore
for i in range(grid_size**2):
    ax = fig.add_subplot(grid_size, grid_size, i+1)  # type: ignore
    ax.axis('off')
    ax.imshow(reconst[i, :,:,0], cmap = 'Greys')
# %%
