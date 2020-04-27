import os
import numpy as np
from devito import configuration
from examples.seismic.skew_self_adjoint import *
configuration['language'] = 'openmp'
configuration['log-level'] = 'DEBUG'

shape = (600, 600, 600)
dtype = np.float32
npad = 10
qmin = 0.1
qmax = 100.0
tmax = 250.0
fpeak = 0.010
omega = 2.0 * np.pi * fpeak

v, b, time_axis, src, rec = defaultSetupIso(npad, shape, dtype, tmax=tmax)
solver = SSA_ISO_AcousticWaveSolver(npad, qmin, qmax, omega, b, v, src, rec, time_axis, space_order=8)
rec, u, summary = solver.forward()
