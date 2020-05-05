import numpy as np
import pytest
from devito import configuration
from examples.seismic import RickerSource
from examples.seismic.skew_self_adjoint import *
configuration['language'] = 'openmp'
configuration['log-level'] = 'ERROR'

# Default values in global scope
npad = 10
fpeak = 0.010
qmin = 0.1
qmax = 500.0
tmax = 1000.0
# shapes = [(181, 161), ]
shapes = [(181, 161), (181, 171, 161)]


class TestWavesolver(object):

#     @pytest.mark.skip(reason="temporarily skip")
    # @pytest.mark.parametrize('shape', [(41, 51), (41, 51, 61)])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('dtype', [np.float64, ])
    @pytest.mark.parametrize('so', [4, ])
    def test_linearity_forward_F(self, shape, dtype, so):
        """
        Test the linearity of the forward modeling operator by verifying:
            a F(s) = F(a s)
        """
        omega = 2.0 * np.pi * fpeak
        b, v, time_axis, src_coords, rec_coords = \
            defaultSetupIso(npad, shape, dtype, tmax=tmax)
        solver = SSA_ISO_AcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                            src_coords, rec_coords, time_axis,
                                            space_order=so)
        src1 = RickerSource(name='src1', grid=v.grid, f0=fpeak, npoint=1,
                            time_range=time_axis)
        src2 = RickerSource(name='src2', grid=v.grid, f0=fpeak, npoint=1,
                            time_range=time_axis)
        src1.coordinates.data[:] = src_coords[:]
        src2.coordinates.data[:] = src_coords[:]
        a = np.random.rand()
        src2.data[:] *= a
        rec1, _, _ = solver.forward(src1)
        rec2, _, _ = solver.forward(src2)
        rec1.data[:] *= a

        # Check receiver wavefeild linearity
        # Normalize by rms of rec2, to enable using abolute tolerance below
        rms2 = np.sqrt(np.mean(rec2.data**2))
        diff = (rec1.data - rec2.data) / rms2
        print("\nlinearity forward F %s rms 1,2,diff; %+12.6e %+12.6e %+12.6e" %
              (shape,
               np.sqrt(np.mean(rec1.data**2)),
               np.sqrt(np.mean(rec2.data**2)),
               np.sqrt(np.mean(diff**2))))
        tol = 1.e-12
        assert np.allclose(diff, 0.0, atol=tol)

#     @pytest.mark.skip(reason="temporarily skip")
    # @pytest.mark.parametrize('shape', [(41, 51), (41, 51, 61)])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('dtype', [np.float64, ])
    @pytest.mark.parametrize('so', [4, ])
    def test_linearity_adjoint_F(self, shape, dtype, so):
        """
        Test the linearity of the adjoint modeling operator by verifying:
            a F^t(r) = F^t(a r)
        """
        omega = 2.0 * np.pi * fpeak
        b, v, time_axis, src_coords, rec_coords = \
            defaultSetupIso(npad, shape, dtype, tmax=tmax)
        solver = SSA_ISO_AcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                            src_coords, rec_coords, time_axis,
                                            space_order=so)
        src0 = RickerSource(name='src0', grid=v.grid, f0=fpeak, npoint=1,
                            time_range=time_axis)
        src0.coordinates.data[:] = src_coords[:]

        rec1 = Receiver(name='rec1', grid=v.grid, time_range=time_axis,
                        coordinates=rec_coords)
        rec2 = Receiver(name='rec2', grid=v.grid, time_range=time_axis,
                        coordinates=rec_coords)
        rec0, _, _ = solver.forward(src0)
        a = np.random.rand()
        rec1.data[:] = rec0.data[:]
        rec2.data[:] = a * rec0.data[:]
        src1, _, _ = solver.adjoint(rec1)
        src2, _, _ = solver.adjoint(rec2)
        src1.data[:] *= a

        # Check adjoint source wavefeild linearity
        # Normalize by rms of rec2, to enable using abolute tolerance below
        rms2 = np.sqrt(np.mean(src2.data**2))
        diff = (src1.data - src2.data) / rms2
        print("\nlinearity adjoint F %s rms 1,2,diff; %+12.6e %+12.6e %+12.6e" %
              (shape,
               np.sqrt(np.mean(src1.data**2)),
               np.sqrt(np.mean(src2.data**2)),
               np.sqrt(np.mean(diff**2))))
        tol = 1.e-12
        assert np.allclose(diff, 0.0, atol=tol)

#     @pytest.mark.skip(reason="temporarily skip")
    # @pytest.mark.parametrize('shape', [(41, 51), (41, 51, 61)])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('dtype', [np.float64, ])
    @pytest.mark.parametrize('so', [4, ])
    def test_adjoint_F(self, shape, dtype, so):
        """
        Test the forward modeling operator by verifying for random s, r:
            r . F(s) = F^t(r) . s
        """
        omega = 2.0 * np.pi * fpeak
        b, v, time_axis, src_coords, rec_coords = \
            defaultSetupIso(npad, shape, dtype, tmax=tmax)
        solver = SSA_ISO_AcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                            src_coords, rec_coords, time_axis,
                                            space_order=8)
        src1 = RickerSource(name='src1', grid=v.grid, f0=fpeak, npoint=1,
                            time_range=time_axis)
        src1.coordinates.data[:] = src_coords[:]

        rec1 = Receiver(name='rec1', grid=v.grid, time_range=time_axis,
                        coordinates=rec_coords)
        rec2, _, _ = solver.forward(src1)
        # flip sign of receiver data for adjoint to make it interesting
        rec1.data[:] = rec2.data[:]
        src2, _, _ = solver.adjoint(rec1)
        sum_s = np.dot(src1.data.reshape(-1), src2.data.reshape(-1))
        sum_r = np.dot(rec1.data.reshape(-1), rec2.data.reshape(-1))
        diff = (sum_s - sum_r) / (sum_s + sum_r)
        print("\nadjoint F %s sum_s, sum_r, diff; %+12.6e %+12.6e %+12.6e" %
              (shape, sum_s, sum_r, diff))
        assert np.isclose(diff, 0., atol=1.e-12)

#     @pytest.mark.skip(reason="temporarily skip")
    # @pytest.mark.parametrize('shape', [(41, 51), (41, 51, 61)])
    @pytest.mark.parametrize('shape', shapes)
    @pytest.mark.parametrize('dtype', [np.float64, ])
    @pytest.mark.parametrize('so', [4, ])
    def test_linearization(self, shape, dtype, so):
        """
        Test the linearization of the forward modeling operator by verifying
        for sequence of h decreasing that the error in the linearization E is
        of second order.

            E = 0.5 || F(m + h   dm) - F(m) - h   J(dm) ||^2

        This is done by fitting a 1st order polynomial to the norms
        """
        omega = 2.0 * np.pi * fpeak
        b, v, time_axis, src_coords, rec_coords = \
            defaultSetupIso(npad, shape, dtype, tmax=tmax)
        solver = SSA_ISO_AcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                            src_coords, rec_coords, time_axis,
                                            space_order=so)
        src = RickerSource(name='src', grid=v.grid, f0=fpeak, npoint=1,
                           time_range=time_axis)
        src.coordinates.data[:] = src_coords[:]

        # Create Functions for models and perturbation
        m0 = Function(name='m0', grid=v.grid, space_order=so)
        mm = Function(name='mm', grid=v.grid, space_order=so)
        dm = Function(name='dm', grid=v.grid, space_order=so)

        # Background model
        m0.data[:] = 1.5

        # Perturbation has magnitude ~ 1 m/s
        # Zero perturbation in the absorbing boundary region
        dm.data[:] = 0
        
        if len(shape) == 2:
            nxp = shape[0] - 2 * npad 
            nzp = shape[1] - 2 * npad 
            dm.data[npad:shape[0]-npad,npad:shape[1]-npad] = \
                -1 + 2 * np.random.rand(nxp, nzp)
        else:
            nxp = shape[0] - 2 * npad 
            nyp = shape[1] - 2 * npad 
            nzp = shape[2] - 2 * npad 
            dm.data[npad:shape[0]-npad, npad:shape[1]-npad, npad:shape[2]-npad] = \
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

        # Alternative linearization test
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
        #     print("h,norm,ratio,exp,|diff|; %+12.6f %+12.6e %+12.6f %+12.6f %+12.6f" %
        #           (h, norms[kstep], dnorm, dh*dh, np.abs(dnorm - dh*dh)))
