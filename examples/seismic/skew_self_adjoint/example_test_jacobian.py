import numpy as np
from devito import configuration
from examples.seismic import RickerSource, Receiver
from examples.seismic.skew_self_adjoint import *
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
mpl.rc('font', size=14)
plt.rcParams['figure.facecolor'] = 'white'

configuration['language'] = 'openmp'
# configuration['log-level'] = 'DEBUG'

nx, ny, nz = 181, 171, 161
shape = (nx, ny, nz)
# nx, nz = 201, 181
# shape = (nx, nz)
dtype = np.float32
npad = 10
qmin = 0.1
qmax = 100.0
tmax = 1000.0
fpeak = 0.010
omega = 2.0 * np.pi * fpeak
space_order = 8

# Model setup
b, v, time_axis, src_coords, rec_coords = \
    defaultSetupIso(npad, shape, dtype, tmax=tmax)
print(time_axis)

# Solver setup
solver = SSA_ISO_AcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                    src_coords, rec_coords, time_axis,
                                    space_order=space_order)

# Source and receiver
src = RickerSource(name='src', grid=v.grid, f0=fpeak, npoint=1, 
                    time_range=time_axis)
src.coordinates.data[:] = src_coords[:]

# Create Functions for models and perturbation
m0 = Function(name='m0', grid=v.grid, space_order=space_order)
mm = Function(name='mm', grid=v.grid, space_order=space_order)
dm = Function(name='dm', grid=v.grid, space_order=space_order)

# Background model
m0.data[:] = 1.5

# Perturbation has magnitude ~ 1 m/s
dm.data[:] = 0
if len(shape) == 2:
    nxp = nx - 2 * npad 
    nzp = nz - 2 * npad 
    dm.data[npad:nx-npad,npad:nz-npad] = -1 + 2 * np.random.rand(nxp, nzp)
else:
    nxp = nx - 2 * npad 
    nyp = ny - 2 * npad 
    nzp = nz - 2 * npad 
    dm.data[npad:nx-npad, npad:ny-npad, npad:nz-npad] = \
        -1 + 2 * np.random.rand(nxp, nyp, nzp)
    
# Compute F(m + dm)
rec0, u0, summary0 = solver.forward(src, v=m0)

# Compute J(dm)
rec1, u1, du, summary1 = solver.jacobian_forward(dm, src=src, v=m0)

# Linearization test via polyfit (see devito/tests/test_gradient.py)
# Solve F(m + h dm) for sequence of decreasing h 
dh = np.sqrt(2.0)
h = 0.1
nstep = 7
scale = np.empty(nstep)
norm1 = np.empty(nstep)
norm2 = np.empty(nstep)
for kstep in range(nstep):
    h = h / dh
    mm.data[:] = m0.data + h * dm.data
    rec2, _, _ = solver.forward(src, v=mm)
    scale[kstep] = h
    norm1[kstep] = 0.5 * np.linalg.norm(rec2.data - rec0.data)**2
    norm2[kstep] = 0.5 * np.linalg.norm(rec2.data - rec0.data - h * rec1.data)**2

# Fit 1st order polynomials to the error sequences
#   Assert the 1st order error has slope dh^2
#   Assert the 2nd order error has slope dh^4
p1 = np.polyfit(np.log10(scale), np.log10(norm1), 1)
p2 = np.polyfit(np.log10(scale), np.log10(norm2), 1)
print('1st order error: %s' % (p1))
print('2nd order error: %s' % (p2))
assert np.isclose(p1[0], dh**2, rtol=0.1)
assert np.isclose(p2[0], dh**4, rtol=0.1)


# Linearization test
# Solve F(m + h dm) for sequence of decreasing h 
# dh = np.sqrt(2.0)
# h = 0.1
# nstep = 10
# norms = np.empty(nstep)
# for kstep in range(nstep):
#     h = h / dh
#     mm.data[:] = m0.data + h * dm.data
#     rec2, _, _ = solver.forward(src, v=mm)
#     d = rec2.data - rec0.data - h * rec1.data
#     norms[kstep] = np.linalg.norm(d.reshape(-1))
#     norms[kstep] = np.sqrt(np.dot(d.reshape(-1), d.reshape(-1)))
#     dnorm = norms[kstep - 1] / norms[kstep] if kstep > 0 else 0
#     print("h,norm,ratio,expected,|diff|; %+12.6f %+12.6e %+12.6f %+12.6f %+12.6f" % 
#           (h, norms[kstep], dnorm, dh*dh, np.abs(dnorm - dh*dh)))

# # Plot
# doPlot = False
# doPlot = True

# if doPlot:
#     h = 0.33
#     dm.data[:] = 0
#     dm.data[nx//2-2:nx//2+5, npad+33-2:npad+33+5] = 1
#     mm.data[:] = m0.data + h * dm.data
#     rec1, u1, du, summary1 = solver.jacobian_forward(dm, src=src, v=m0)
#     rec2, u2, summary2 = solver.forward(src, v=mm)

#     print("")
#     print("m0            min/max; %+12.6f %+12.6f" % (np.min(m0.data), np.max(m0.data)))
#     print("dm            min/max; %+12.6f %+12.6f" % (np.min(dm.data), np.max(dm.data)))
#     print("h * dm        min/max; %+12.6f %+12.6f" % (np.min(h * dm.data), np.max(h * dm.data)))
#     print("m0 + h dm     min/max; %+12.6f %+12.6f" % (np.min(mm.data), np.max(mm.data)))

#     print("")
#     print("F(m+ h dm)    min/max; %+12.6f %+12.6f" % (np.min(rec2.data), np.max(rec2.data)))
#     print("F(m)          min/max; %+12.6f %+12.6f" % (np.min(rec0.data), np.max(rec0.data)))
#     print("J(dm)         min/max; %+12.6f %+12.6f" % (np.min(rec1.data), np.max(rec1.data)))
#     print("F(m)+ h J(dm) min/max; %+12.6f %+12.6f" % 
#           (np.min(rec0.data + h * rec1.data), np.max(rec0.data + h * rec1.data)))
#     print("Difference    min/max; %+12.6f %+12.6f" % 
#           (np.min(rec2.data - rec0.data - h * rec1.data), np.max(rec2.data - rec0.data - h * rec1.data)))

#     amaxA = 0.05 * np.max(np.abs(rec2.data))

#     plt.figure(figsize=(25,10))
#     plt.subplot(1, 5, 1)
#     plt.imshow(rec2.data, cmap="gray", 
#                vmin=-amaxA, vmax=amaxA, aspect="auto")
#     plt.colorbar(orientation='horizontal', label='Amplitude')
#     plt.xlabel("X Coordinate (m)")
#     plt.ylabel("Z Coordinate (m)")
#     plt.title("$F(m + h\ \delta m)$")

#     plt.subplot(1, 5, 2)
#     plt.imshow(rec0.data, cmap="gray", 
#                vmin=-amaxA, vmax=amaxA, aspect="auto")
#     plt.colorbar(orientation='horizontal', label='Amplitude')
#     plt.xlabel("X Coordinate (m)")
#     plt.ylabel("Z Coordinate (m)")
#     plt.title("$F(m)$")

#     plt.subplot(1, 5, 3)
#     plt.imshow(rec0.data + h * rec1.data, cmap="gray", 
#                vmin=-amaxA, vmax=amaxA, aspect="auto")
#     plt.colorbar(orientation='horizontal', label='Amplitude')
#     plt.xlabel("X Coordinate (m)")
#     plt.ylabel("Z Coordinate (m)")
#     plt.title("$F(m) + h\ J(\delta m)$")

#     plt.subplot(1, 5, 4)
#     plt.imshow(h * rec1.data, cmap="gray", 
#                vmin=-amaxA/10, vmax=amaxA/10, aspect="auto")
#     plt.colorbar(orientation='horizontal', label='Amplitude x10')
#     plt.xlabel("X Coordinate (m)")
#     plt.ylabel("Z Coordinate (m)")
#     plt.title("$h\ J(\delta m)$")

#     plt.subplot(1, 5, 5)
#     plt.imshow(rec2.data - rec0.data - h * rec1.data, cmap="gray", 
#                vmin=-amaxA/10, vmax=amaxA/10, aspect="auto")
#     plt.colorbar(orientation='horizontal', label='Amplitude x10')
#     plt.xlabel("X Coordinate (m)")
#     plt.ylabel("Z Coordinate (m)")
#     plt.title("$F(m + h\ dm) - F(m) - h\ J(\delta m)$")

#     plt.tight_layout()
#     plt.savefig("linearization.png")
#     plt.show()
