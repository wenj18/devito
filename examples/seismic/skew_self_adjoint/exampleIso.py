import numpy as np
from devito import configuration
from examples.seismic import RickerSource, Receiver, TimeAxis
from examples.seismic.skew_self_adjoint import *
configuration['language'] = 'openmp'
configuration['log-level'] = 'ERROR'
# configuration['mpi'] = 1
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm


# shape = (601, 601, 601)
shape = (201, 201)
#dtype = np.float32
dtype = np.float64
npad = 20
qmin = 0.1
qmax = 1000.0
tmax = 1000.0
fpeak = 0.010
omega = 2.0 * np.pi * fpeak

b, v, time_axis, src, rec = defaultSetupIso(npad, shape, dtype, tmax=tmax)
solver = SSA_ISO_AcousticWaveSolver(npad, qmin, qmax, omega, b, v,
                                    src, rec, time_axis, space_order=8)
nt,ns = src.data.shape
nt,nr = rec.data.shape
print("ns, nr; ", ns, nr)
print(time_axis)

print("v.grid.shape;     ", v.grid.shape)
print("v.grid.origin;    ", v.grid.origin)
print("v.grid.origin[0]; ", v.grid.origin[0])
print("v.grid.origin[1]; ", v.grid.origin[1])
print("v.grid.spacing;   ", v.grid.spacing)
print("v min/max;        ", np.min(v.data), np.max(v.data))
print("b min/max;        ", np.min(b.data), np.max(b.data))
print("rec size;         ", rec.coordinates.data.shape)
print("rec first;        ", rec.coordinates.data[0, 0], rec.coordinates.data[0, 1])
print("rec last;         ", rec.coordinates.data[-1, 0], rec.coordinates.data[-1, 1])

testForwardLinearity = True
testAdjointLinearity = True
doPlot = True
doPlot = False

if testForwardLinearity:
    tol = 1.e-12
    print("")
    print("Test forward linearity -- tol (%.2e)" % (tol))
    a = 2.5
    src1 = RickerSource(name='src1', grid=v.grid, f0=fpeak, npoint=1, time_range=time_axis)
    src2 = RickerSource(name='src2', grid=v.grid, f0=fpeak, npoint=1, time_range=time_axis)
    src1.coordinates.data[:] = src.coordinates.data[:] 
    src2.coordinates.data[:] = src.coordinates.data[:] 
    src2.data[:] *= a
    rec1, _, _ = solver.forward(src1)
    rec2, _, _ = solver.forward(src2)
    rec1.data[:] *= a

    # normalize by rms of rec2, to enable using abolute tolerance below
    rms2 = np.sqrt(np.mean(rec2.data**2))
    rec1.data[:] = rec1.data[:] / rms2
    rec2.data[:] = rec2.data[:] / rms2
    diff = rec1.data - rec2.data

    min1, max1, rms1 = np.min(rec1.data), np.max(rec1.data), np.sqrt(np.mean(rec1.data**2))
    min2, max2, rms2 = np.min(rec2.data), np.max(rec2.data), np.sqrt(np.mean(rec2.data**2))
    mind, maxd, rmsd = np.min(np.abs(diff)), np.max(np.abs(diff)), np.sqrt(np.mean(diff**2))

    print("rec1 min,max,rms; %+12.6e %+12.6e %+12.6e" % (min1, max1, rms1))
    print("rec2 min,max,rms; %+12.6e %+12.6e %+12.6e" % (min2, max2, rms2))
    print("diff min,max,rms; %+12.6e %+12.6e %+12.6e" % (mind, maxd, rmsd))
    
    s = np.flip(np.sort(diff[:], axis=None), axis=None)
    for i in range(30):
        print("k,diff,tol; %3d %+12.6e %+12.6e" % (i, s[i], tol))
    
    print("nr,nt,diff.shape; ", nr, nt, diff.shape)

    assert np.allclose(diff, 0.0, atol=tol)
    
    if doPlot:
        amax = np.max(np.abs(rec1.data)) / 5
        origin = v.grid.origin
        spacing = v.grid.spacing
        plt_extent = [origin[0], origin[0] + spacing[0]*(shape[0]-1),
                      origin[1] + spacing[1]*(shape[1]-1), origin[1]]
        plt.figure(figsize=(16,12))

        plt.subplot(1, 3, 1)
        plt.imshow(rec1.data / amax, cmap="seismic", vmin=-1, vmax=+1, aspect="auto")
        plt.xlabel("a F(s)")
        plt.ylabel("Time (msec)")
        plt.title("Rec Gather 1")

        plt.subplot(1, 3, 2)
        plt.imshow(rec2.data / amax, cmap="seismic", vmin=-1, vmax=+1, aspect="auto")
        plt.title("Rec Gather 2")
        plt.xlabel("F(a s)")
        plt.ylabel("Time (msec)")

        plt.subplot(1, 3, 3)
        plt.imshow(1.e14 * diff / amax, cmap="seismic", vmin=-1, vmax=+1, aspect="auto")
        plt.title("1.e14 X Diff")
        plt.xlabel("Receiver")
        plt.ylabel("Time (msec)")

        plt.tight_layout()
        plt.show()

# , extent=plt_extent


# if testAdjointLinearity:
#     print("")
#     print("Test adjoint linearity ...")
#     a = np.random.rand()
#     rec1 = Receiver(name='rec1', grid=v.grid, time_range=time_axis,
#                     coordinates=rec.coordinates)
#     rec2 = Receiver(name='rec2', grid=v.grid, time_range=time_axis,
#                     coordinates=rec.coordinates)
#     rec2.data[:] *= a
#     src1, _, _ = solver.adjoint(rec1)
#     rec2, _, _ = solver.forward(src2)
#     rec1.data[:] *= a
#     diff = rec1.data - rec2.data
#     rms1 = np.sqrt(np.mean(rec1.data**2))
#     rms2 = np.sqrt(np.mean(rec1.data**2))
#     rmsd = np.sqrt(np.mean(diff**2))
#     print("rms1; %+12.6e" % rms1)
#     print("rms2; %+12.6e" % rms2)
#     print("rmsd; %+12.6e" % rmsd)
#     assert np.allclose(rec1.data[:], rec2.data[:], rtol=0, atol=tol)
    
