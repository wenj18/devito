import numpy as np
from devito import (Grid, Constant, Function, SpaceDimension, configuration)
from examples.seismic.skew_self_adjoint import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
# %matplotlib inline
mpl.rc('font', size=14)
plt.rcParams['figure.facecolor'] = 'white'
configuration['language'] = 'openmp'
configuration['log-level'] = 'DEBUG'

shape = (600, 600)
dtype = np.float32
npad = 50
qmin = 0.1
qmax = 100.0
fpeak = 0.010
w = 2.0 * np.pi * fpeak
d = 10.0
origin = tuple([0.0 - d * npad for s in shape])
spacing = (d, d)
extent = tuple([o + d * (s - 1) for o, s in zip(origin, shape)])
x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=d))
z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=d))
grid = Grid(extent=extent, shape=shape, origin=origin, dimensions=(x, z), dtype=dtype)

print("shape;      ", shape)
print("origin;     ", origin)
print("spacing;    ", spacing)
print("extent;     ", extent)

wOverQ_a = Function(name='wOverQ', grid=grid)
wOverQ_b = Function(name='wOverQ', grid=grid)

sigma = 0
setup_wOverQ_numpy(wOverQ_a, w, qmin, qmax, npad, sigma=sigma)
setup_wOverQ(wOverQ_b, w, qmin, qmax, npad, sigma=sigma)

# Plot
qa = np.log10(w / wOverQ_a.data)
qb = np.log10(w / wOverQ_b.data)
lmin, lmax = np.log10(qmin), np.log10(qmax)

# qa = wOverQ_a.data
# qb = wOverQ_b.data
# lmin, lmax = np.min(wOverQ_a.data), np.max(wOverQ_a.data)

plt_extent = [origin[0], origin[0] + spacing[0]*(shape[0]-1),
              origin[1] + spacing[1]*(shape[1]-1), origin[1]]

print("plt_extent; ", plt_extent)

fig, ax = plt.subplots(1, 2, figsize=(12, 8))
plt.subplots_adjust(left=0.05, right=0.95, wspace=0.25)

plt.subplot(1, 2, 1)
im025 = plt.imshow(np.transpose(qa.data), cmap=cm.jet_r,
                   vmin=lmin, vmax=lmax, extent=plt_extent)
plt.plot([origin[0], origin[0], extent[0], extent[0], origin[0]],
         [origin[1], extent[1], extent[1], origin[1], origin[1]],
         'white', linewidth=4, linestyle=':', label="Absorbing Boundary")
plt.xlabel("X Coordinate (m)", labelpad=15)
plt.ylabel("Z Coordinate (m)", labelpad=15)
plt.title("log10 of wOverQ_a model", y=1.035)

plt.subplot(1, 2, 2)
im100 = plt.imshow(np.transpose(qb.data), cmap=cm.jet_r,
                   vmin=lmin, vmax=lmax, extent=plt_extent)
plt.plot([origin[0], origin[0], extent[0], extent[0], origin[0]],
         [origin[1], extent[1], extent[1], origin[1], origin[1]],
         'white', linewidth=4, linestyle=':', label="Absorbing Boundary")
plt.xlabel("X Coordinate (m)", labelpad=15)
plt.ylabel("Z Coordinate (m)", labelpad=15)
plt.title("log10 of wOverQ_b model", y=1.025)

plt.draw()
p0 = ax[0].get_position().get_points().flatten()
p1 = ax[1].get_position().get_points().flatten()
print("p0; ", p0)
print("p1; ", p1)
ax_cbar = fig.add_axes([p0[0], 0, p1[2]-p0[0], 0.05])
cbar = plt.colorbar(im100, cax=ax_cbar, orientation='horizontal')
cbar.set_label('Log10 of Q(x,z)', labelpad=-75, y=1.035, rotation=0)

plt.show()
