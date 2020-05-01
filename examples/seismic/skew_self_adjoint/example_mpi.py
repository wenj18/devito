import numpy as np
from devito import configuration
from examples.seismic import RickerSource, Receiver
from examples.seismic.skew_self_adjoint import *

configuration['language'] = 'openmp'
configuration['log-level'] = 'DEBUG'
configuration['mpi'] = 1

shape = (601, 601, 601)
dtype = np.float32
npad = 20
qmin = 0.1
qmax = 1000.0
tmax = 1000.0
fpeak = 0.010
omega = 2.0 * np.pi * fpeak

b, v, time_axis, src, rec = defaultSetupIso(npad, shape, dtype, tmax=tmax)
solver = SSA_ISO_AcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                    src, rec, time_axis, space_order=8)
tol = 1.e-12
a = np.random.rand()
rec, _, _ = solver.forward(src)

