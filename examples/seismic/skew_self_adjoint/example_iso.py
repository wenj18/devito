import numpy as np
from devito import configuration
from examples.seismic import RickerSource, Receiver
from examples.seismic.skew_self_adjoint import *
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from matplotlib import cm

configuration['language'] = 'openmp'
configuration['log-level'] = 'ERROR'
# configuration['log-level'] = 'DEBUG'
# configuration['mpi'] = 1


# shape = (601, 601, 601)
shape = (201, 201)
# dtype = np.float32
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
nt, ns = src.data.shape
nt, nr = rec.data.shape
print(time_axis)
print("ns, nr; ", ns, nr)
print("grid.shape;     ", v.grid.shape)
print("grid.origin;    ", v.grid.origin)
print("grid.origin[0]; ", v.grid.origin[0].data)
print("grid.origin[1]; ", v.grid.origin[1].data)
print("grid.spacing;   ", v.grid.spacing)

testForwardLinearity = True
testForwardLinearity = False
testAdjointLinearity = True
doPlot = True
doPlot = False

if testForwardLinearity:
    tol = 1.e-12
    a = np.random.rand()
    print("")
    print("Test forward linearity -- tol (%.2e) -- a (%+f)" % (tol, a))
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

    # normalize by rms of rec2, to enable using abolute tolerance below
    rms2 = np.sqrt(np.mean(rec2.data**2))
    rec1.data[:] = rec1.data[:] / rms2
    rec2.data[:] = rec2.data[:] / rms2
    diff = rec1.data - rec2.data

    print("rec1 min,max,rms; %+12.6e %+12.6e %+12.6e" % 
        (np.min(rec1.data), np.max(rec1.data), np.sqrt(np.mean(rec1.data**2))))
    print("rec2 min,max,rms; %+12.6e %+12.6e %+12.6e" % 
        (np.min(rec2.data), np.max(rec2.data), np.sqrt(np.mean(rec2.data**2))))
    print("diff min,max,rms; %+12.6e %+12.6e %+12.6e" % 
        (np.min(diff), np.max(diff), np.sqrt(np.mean(diff**2))))

    assert np.allclose(diff, 0.0, atol=tol)

    if doPlot:
        amax = np.max(np.abs(rec1.data)) / 5
        origin = v.grid.origin
        spacing = v.grid.spacing
        plt_extent = [origin[0], origin[0] + spacing[0]*(shape[0]-1),
                      origin[1] + spacing[1]*(shape[1]-1), origin[1]]
        plt.figure(figsize=(16, 12))

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

if testAdjointLinearity:
    tol = 1.e-12
    a = np.random.rand()
    print("")
    print("Test adjoint linearity -- tol (%.2e) -- a (%+f)" % (tol, a))
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

    print("")
    print("src0 min,max,rms; %+12.6e %+12.6e %+12.6e" % 
        (np.min(src0.data), np.max(src0.data), np.sqrt(np.mean(src0.data**2))))
    print("rec0 min,max,rms; %+12.6e %+12.6e %+12.6e" % 
        (np.min(rec0.data), np.max(rec0.data), np.sqrt(np.mean(rec0.data**2))))
    print("rec1 min,max,rms; %+12.6e %+12.6e %+12.6e" % 
        (np.min(rec1.data), np.max(rec1.data), np.sqrt(np.mean(rec1.data**2))))
    print("rec2 min,max,rms; %+12.6e %+12.6e %+12.6e" % 
        (np.min(rec2.data), np.max(rec2.data), np.sqrt(np.mean(rec2.data**2))))

    
    src1, u1, _ = solver.adjoint(rec1)
    print("")
    print("src1 min,max,rms; %+12.6e %+12.6e %+12.6e" % 
        (np.min(src1.data), np.max(src1.data), np.sqrt(np.mean(src1.data**2))))
    print("rec1 min,max,rms; %+12.6e %+12.6e %+12.6e" % 
        (np.min(rec1.data), np.max(rec1.data), np.sqrt(np.mean(rec1.data**2))))

    
    src2, u2, _ = solver.adjoint(rec2)
    print("")
    print("src2 min,max,rms; %+12.6e %+12.6e %+12.6e" % 
        (np.min(src2.data), np.max(src2.data), np.sqrt(np.mean(src2.data**2))))
    print("rec2 min,max,rms; %+12.6e %+12.6e %+12.6e" % 
        (np.min(rec2.data), np.max(rec2.data), np.sqrt(np.mean(rec2.data**2))))
    
    print("")
    print("src0 xmin,xmax,zmin,zmax; %+12.6f %+12.6f %+12.6f %+12.6f" % 
        (np.min(src0.coordinates.data[:,0]), np.max(src0.coordinates.data[:,0]), 
         np.min(src0.coordinates.data[:,1]), np.max(src0.coordinates.data[:,1])))
    print("src1 xmin,xmax,zmin,zmax; %+12.6f %+12.6f %+12.6f %+12.6f" % 
        (np.min(src1.coordinates.data[:,0]), np.max(src1.coordinates.data[:,0]), 
         np.min(src1.coordinates.data[:,1]), np.max(src1.coordinates.data[:,1])))
    print("src2 xmin,xmax,zmin,zmax; %+12.6f %+12.6f %+12.6f %+12.6f" % 
        (np.min(src2.coordinates.data[:,0]), np.max(src2.coordinates.data[:,0]), 
         np.min(src2.coordinates.data[:,1]), np.max(src2.coordinates.data[:,1])))

    print("")
    print("rec  xmin,xmax,zmin,zmax; %+12.6f %+12.6f %+12.6f %+12.6f" % 
        (np.min(rec.coordinates.data[:,0]), np.max(rec.coordinates.data[:,0]), 
         np.min(rec.coordinates.data[:,1]), np.max(rec.coordinates.data[:,1])))
    print("rec1 xmin,xmax,zmin,zmax; %+12.6f %+12.6f %+12.6f %+12.6f" % 
        (np.min(rec1.coordinates.data[:,0]), np.max(rec1.coordinates.data[:,0]), 
         np.min(rec1.coordinates.data[:,1]), np.max(rec1.coordinates.data[:,1])))
    print("rec2 xmin,xmax,zmin,zmax; %+12.6f %+12.6f %+12.6f %+12.6f" % 
        (np.min(rec2.coordinates.data[:,0]), np.max(rec2.coordinates.data[:,0]), 
         np.min(rec2.coordinates.data[:,1]), np.max(rec2.coordinates.data[:,1])))

    
    src1.data[:] *= a

    # normalize by rms of rec2, to enable using abolute tolerance below
    rms2 = np.sqrt(np.mean(src2.data**2))
    src1.data[:] = src1.data[:] / rms2
    src2.data[:] = src2.data[:] / rms2
    diff = src1.data - src2.data

    print("diff min,max,rms; %+12.6e %+12.6e %+12.6e" % 
        (np.min(diff), np.max(diff), np.sqrt(np.mean(diff**2))))

    assert np.allclose(diff, 0.0, atol=tol)
    
#     if doPlot:
#         amax = np.max(np.abs(rec1.data)) / 5
#         origin = v.grid.origin
#         spacing = v.grid.spacing
#         plt_extent = [origin[0], origin[0] + spacing[0]*(shape[0]-1),
#                       origin[1] + spacing[1]*(shape[1]-1), origin[1]]
#         plt.figure(figsize=(16,12))

#         plt.subplot(1, 3, 1)
#         plt.imshow(rec1.data / amax, cmap="seismic", vmin=-1, vmax=+1, aspect="auto")
#         plt.xlabel("a F(s)")
#         plt.ylabel("Time (msec)")
#         plt.title("Rec Gather 1")

#         plt.subplot(1, 3, 2)
#         plt.imshow(rec2.data / amax, cmap="seismic", vmin=-1, vmax=+1, aspect="auto")
#         plt.title("Rec Gather 2")
#         plt.xlabel("F(a s)")
#         plt.ylabel("Time (msec)")

#         plt.subplot(1, 3, 3)
#         plt.imshow(1.e14 * diff / amax, cmap="seismic", vmin=-1, vmax=+1, aspect="auto")
#         plt.title("1.e14 X Diff")
#         plt.xlabel("Receiver")
#         plt.ylabel("Time (msec)")

#         plt.tight_layout()
#         plt.show()

