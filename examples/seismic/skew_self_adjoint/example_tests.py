import numpy as np
from devito import configuration
from examples.seismic import RickerSource, Receiver
from examples.seismic.skew_self_adjoint import *

configuration['language'] = 'openmp'
# configuration['log-level'] = 'DEBUG'

# nx, ny, nz = 889, 889, 379
# shape = (nx, ny, nz)
nx, nz = 501, 501
shape = (nx, nz)
dtype = np.float32
npad = 50
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
dm.data[npad:nx-npad, npad:nz-npad] = -1 + 2 * np.random.rand(*shape) / 1000

# Compute F(m + dm)
rec0, _, _ = solver.forward(src, v=m0)

# Compute J(dm)
rec1, _, _, _ = solver.jacobian_forward(dm, src=src, v=m0)

# Solve F(m + h dm) for sequence of decreasing h 
# dh = np.sqrt(2.0)
dh = 2.0
h = 100.0
nstep = 8
norms = np.empty(nstep)
for kstep in range(nstep):
    mm.data[:] = m0.data[:] + h * dm.data[:]
    rec2, _, _ = solver.forward(src, v=mm)
    norms[kstep] = np.linalg.norm(rec2.data[:] - rec0.data[:] - h * rec1.data[:])
    dnorm = norms[kstep - 1] / norms[kstep] if kstep > 0 else 0.0
    print("h,dm,norm; %+12.6f %+12.6f %+12.6e %+12.6f" % 
          (h, np.max(np.abs(mm.data[:] - m0.data[:])), norms[kstep], dnorm))
    h = h / dh
          
#tol = 1.e-12
#a = np.random.rand()
