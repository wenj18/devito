import os
import numpy as np
from devito import Grid, Function, Eq, Operator, configuration
from examples.seismic.skew_self_adjoint import *
configuration['language'] = 'openmp'
configuration['log-level'] = 'DEBUG'
# configuration['log-level'] = 'PERF'
# configuration['autotuning'] = 'aggressive'
# configuration['first-touch'] = 1
configuration['mpi'] = 1
# configuration['mpi'] = 'full'

# os.environ["OMP_PLACES"] = 'cores'
# os.environ["OMP_PROC_BIND"] = 'spread'

# shape = (600, 600)
# shape = (600, 600, 600)
shape = (881, 881, 371)
dtype = np.float32
npad = 10
qmin = 0.1
qmax = 100.0
tmax = 250.0
fpeak = 0.010
omega = 2.0 * np.pi * fpeak

v, b, time_axis, src, rec = defaultSetupIso(npad, shape, dtype, tmax=tmax)
print(time_axis)
# solver = SSA_ISO_AcousticWaveSolver(npad, qmin, qmax, omega, b, v, src, rec, time_axis, space_order=4)
solver = SSA_ISO_AcousticWaveSolver(npad, qmin, qmax, omega, b, v, src, rec, time_axis, space_order=8)
# solver = SSA_ISO_AcousticWaveSolver(npad, qmin, qmax, omega, b, v, src, rec, time_axis, space_order=16)
rec, u, summary = solver.forward()
# rec, u, summary = solver.forward(autotune=('aggressive', 'runtime'))
