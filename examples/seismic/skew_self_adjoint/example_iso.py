import numpy as np
from devito import configuration
from examples.seismic import RickerSource, Receiver
from examples.seismic.skew_self_adjoint import *

configuration['language'] = 'openmp'
# configuration['log-level'] = 'ERROR'
configuration['log-level'] = 'DEBUG'
# configuration['debug_compiler'] = 1
# configuration['log-level'] = 'PERF'
# configuration['mpi'] = 1
# configuration['mpi'] = 'full'
# configuration['autotuning'] = ("aggressive", 'preemptive') 
# configuration['autotuning'] = "aggressive"

shape = (601, 601, 601)
# shape = (601, 601)
dtype = np.float32
# dtype = np.float64
npad = 20
qmin = 0.1
qmax = 1000.0
tmax = 500.0
fpeak = 0.010
omega = 2.0 * np.pi * fpeak

b, v, time_axis, src_coords, rec_coords = defaultSetupIso(npad, shape, dtype, tmax=tmax)

solver = SSA_ISO_AcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                    src_coords, rec_coords, time_axis, space_order=8)

src = RickerSource(name='src', grid=v.grid, f0=fpeak, npoint=1, time_range=time_axis)
src.coordinates.data[:] = src_coords[:]

ns = src_coords.shape[0]
nr = rec_coords.shape[0]
print(time_axis)
print("ns, nr;         ", ns, nr)
print("grid.shape;     ", v.grid.shape)
print("grid.origin;    ", (v.grid.origin[0].data, v.grid.origin[1].data))
print("grid.spacing;   ", v.grid.spacing)

tol = 1.e-12
a = np.random.rand()
rec, _, _ = solver.forward(src)

