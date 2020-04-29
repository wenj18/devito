import numpy as np
import pytest
from devito import configuration
from examples.seismic import RickerSource
from examples.seismic.skew_self_adjoint import *
configuration['language'] = 'openmp'
configuration['log-level'] = 'ERROR'

# Default values in global scope
npad = 20
fpeak = 0.010
qmin = 0.1
qmax = 500.0
tmax = 1000.0


class TestWavesolver(object):

    # @pytest.mark.skip(reason="temporarily skip")
    # @pytest.mark.parametrize('shape', [(41, 51), (41, 51, 61)])
    @pytest.mark.parametrize('shape', [(201, 201), ])
    @pytest.mark.parametrize('dtype', [np.float64, ])
    @pytest.mark.parametrize('so', [4, ])
    def test_linearity_F(self, shape, dtype, so):
        """
        Test the linearity of the modeling operator by verifying:
            a F(s) = F(a s)
        """
        tol = 1.e-12
        omega = 2.0 * np.pi * fpeak

        b, v, time_axis, src, rec = defaultSetupIso(npad, shape, dtype, tmax=tmax)
        solver = SSA_ISO_AcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                            src, rec, time_axis, space_order=so)
        nt, ns = src.data.shape
        nt, nr = rec.data.shape
        print("ns, nr; ", ns, nr)
        print(time_axis)

        a = 2.5
        src1 = RickerSource(name='src1', grid=v.grid, f0=fpeak, npoint=1,
                            time_range=time_axis)
        src2 = RickerSource(name='src2', grid=v.grid, f0=fpeak, npoint=1,
                            time_range=time_axis)
        src1.coordinates.data[:] = src.coordinates.data[:]
        src2.coordinates.data[:] = src.coordinates.data[:]
        src2.data[:] *= a
        rec1, _, _ = solver.forward(src1)
        rec2, _, _ = solver.forward(src2)
        rec1.data[:] *= a

        # normalize by rms of rec2, to enable abolute tolerance below
        rms2 = np.sqrt(np.mean(rec2.data**2))
        rec1.data[:] = rec1.data[:] / rms2
        rec2.data[:] = rec2.data[:] / rms2
        diff = rec1.data - rec2.data
        mind, maxd, rmsd = np.min(np.abs(diff)), np.max(np.abs(diff)), np.sqrt(np.mean(diff**2))
        print("diff min,max,rms; %+12.6e %+12.6e %+12.6e" % (mind, maxd, rmsd))
        assert np.allclose(diff, 0.0, rtol=0, atol=tol)
