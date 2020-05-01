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
    def test_linearity_forward_F(self, shape, dtype, so):
        """
        Test the linearity of the forward modeling operator by verifying:
            a F(s) = F(a s)
        """
        omega = 2.0 * np.pi * fpeak
        b, v, time_axis, src, rec = defaultSetupIso(npad, shape, dtype, tmax=tmax)
        solver = SSA_ISO_AcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                            src, rec, time_axis, space_order=8)
        tol = 1.e-12
        a = np.random.rand()
        
        # Set up forward problems
        src1 = RickerSource(name='src1', grid=v.grid, f0=fpeak, npoint=1,
                            time_range=time_axis)
        src2 = RickerSource(name='src2', grid=v.grid, f0=fpeak, npoint=1,
                            time_range=time_axis)
        src1.coordinates.data[:] = src.coordinates.data[:]
        src2.coordinates.data[:] = src.coordinates.data[:]
        
        # Solve forward problems
        src2.data[:] *= a
        rec1, _, _ = solver.forward(src1)
        rec2, _, _ = solver.forward(src2)
        rec1.data[:] *= a

        # Check receiver wavefeild linearity
        # Normalize by rms of rec2, to enable using abolute tolerance below
        rms2 = np.sqrt(np.mean(rec2.data**2))
        diff = (rec1.data - rec2.data) / rms2
        print("rec1 min,max,rms; %+12.6e %+12.6e %+12.6e" % 
            (np.min(rec1.data), np.max(rec1.data), np.sqrt(np.mean(rec1.data**2))))
        print("rec2 min,max,rms; %+12.6e %+12.6e %+12.6e" % 
            (np.min(rec2.data), np.max(rec2.data), np.sqrt(np.mean(rec2.data**2))))
        print("diff min,max,rms; %+12.6e %+12.6e %+12.6e" % 
            (np.min(diff), np.max(diff), np.sqrt(np.mean(diff**2))))
        assert np.allclose(diff, 0.0, atol=tol)

    # @pytest.mark.parametrize('shape', [(41, 51), (41, 51, 61)])
    @pytest.mark.parametrize('shape', [(201, 201), ])
    @pytest.mark.parametrize('dtype', [np.float64, ])
    @pytest.mark.parametrize('so', [4, ])
    def test_linearity_adjoint_F(self, shape, dtype, so):
        """
        Test the linearity of the adjoint modeling operator by verifying:
            a F^t(r) = F^t(a r)
        """
        omega = 2.0 * np.pi * fpeak
        b, v, time_axis, src, rec = defaultSetupIso(npad, shape, dtype, tmax=tmax)
        solver = SSA_ISO_AcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                            src, rec, time_axis, space_order=8)
        tol = 1.e-12
        a = np.random.rand()
        
        # Set up adjoint problems
        src0 = RickerSource(name='src0', grid=v.grid, f0=fpeak, npoint=1,
                            time_range=time_axis)
        src0.coordinates.data[:] = src.coordinates.data[:]

        rec1 = Receiver(name='rec1', grid=v.grid, time_range=time_axis,
                        coordinates=rec.coordinates)
        rec2 = Receiver(name='rec2', grid=v.grid, time_range=time_axis,
                        coordinates=rec.coordinates)

        # Solve forward problems to generate receiver wavefield
        rec0, _, _ = solver.forward(src0)

        # Solve adjoint problems
        rec1.data[:] = rec0.data[:]
        rec2.data[:] = a * rec0.data[:]
        src1, _, _ = solver.adjoint(rec1)
        src2, _, _ = solver.adjoint(rec2)
        src1.data[:] *= a

        # Check adjoint source wavefeild linearity
        # Normalize by rms of rec2, to enable using abolute tolerance below
        rms2 = np.sqrt(np.mean(src2.data**2))
        diff = (src1.data - src2.data) / rms2
        print("src1 min,max,rms; %+12.6e %+12.6e %+12.6e" % 
            (np.min(src1.data), np.max(src1.data), np.sqrt(np.mean(src1.data**2))))
        print("src2 min,max,rms; %+12.6e %+12.6e %+12.6e" % 
            (np.min(src2.data), np.max(src2.data), np.sqrt(np.mean(src2.data**2))))
        print("diff min,max,rms; %+12.6e %+12.6e %+12.6e" % 
            (np.min(diff), np.max(diff), np.sqrt(np.mean(diff**2))))
        assert np.allclose(diff, 0.0, atol=tol)
