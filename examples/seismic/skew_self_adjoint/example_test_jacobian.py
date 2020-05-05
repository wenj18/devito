import numpy as np
from devito import configuration
from examples.seismic import RickerSource, Receiver
from examples.seismic.skew_self_adjoint import *

configuration['language'] = 'openmp'
# configuration['log-level'] = 'DEBUG'

# nx, ny, nz = 181, 171, 161
# shape = (nx, ny, nz)
nx, nz = 201, 181
shape = (nx, nz)
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

# Source 
src0 = RickerSource(name='src0', grid=v.grid, f0=fpeak, npoint=1, 
                    time_range=time_axis)
src0.coordinates.data[:] = src_coords[:]

# Create Functions for models and perturbations
m0 = Function(name='m0', grid=v.grid, space_order=space_order)
m1 = Function(name='m1', grid=v.grid, space_order=space_order)

# Background model
m0.data[:] = 1.5

# Model perturbation
m1.data[:] = 0
if len(shape) == 2:
    nxp = nx - 2 * npad 
    nzp = nz - 2 * npad 
    m1.data[npad:nx-npad,npad:nz-npad] = -1 + 2 * np.random.rand(nxp, nzp)
else:
    nxp = nx - 2 * npad 
    nyp = ny - 2 * npad 
    nzp = nz - 2 * npad 
    m1.data[npad:nx-npad, npad:ny-npad, npad:nz-npad] = \
        -1 + 2 * np.random.rand(nxp, nyp, nzp)

# Data perturbation
rec1 = Receiver(name='rec1', grid=v.grid, time_range=time_axis,
                coordinates=rec_coords)
nt,nr = rec1.data.shape
rec1.data[:] = np.random.rand(nt, nr) 
print("rec1.shape; ", rec1.shape)
print("nr,nt; ", nr, nt)
print(time_axis)

rec2, u0, _, _ = solver.jacobian_forward(m1, src0, v=m0, save=nt)
m2, _, _, _ = solver.jacobian_adjoint(rec1, u0, v=m0, save=nt)

#         sum_s = np.dot(src1.data.reshape(-1), src2.data.reshape(-1))
#         sum_r = np.dot(rec1.data.reshape(-1), rec2.data.reshape(-1))
#         diff = (sum_s - sum_r) / (sum_s + sum_r)
#         print("\nadjoint F %s sum_s, sum_r, diff; %+12.6e %+12.6e %+12.6e" %
#               (shape, sum_s, sum_r, diff))
#         assert np.isclose(diff, 0., atol=1.e-12)
