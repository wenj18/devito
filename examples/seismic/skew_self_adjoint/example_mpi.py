import numpy as np
from devito import (configuration, SpaceDimension, Constant, Grid,
                    Operator)
from examples.seismic import RickerSource, Receiver, TimeAxis
from examples.seismic.skew_self_adjoint import *

configuration['language'] = 'openmp'
configuration['log-level'] = 'DEBUG'

# Constants
nx, ny, nz = 889, 889, 379
shape = (nx, ny, nz)
space_order = 8
d = 10.0
dt = 0.5
qmin = 0.1
qmax = 1000.0
tmax = 500.0
fpeak = 0.010
omega = 2.0 * np.pi * fpeak

# Dimensions, Grid, velocity Function
dtype = np.float32
origin = (0.0, 0.0, 0.0)
extent = (d * (nx-1), d * (ny - 1), d * (nz - 1))
print("origin; ", origin)
print("extent; ", extent)
x = SpaceDimension(name='x', spacing=Constant(name='h_x', value=d))
y = SpaceDimension(name='y', spacing=Constant(name='h_y', value=d))
z = SpaceDimension(name='z', spacing=Constant(name='h_z', value=d))
grid = Grid(extent=extent, shape=shape, origin=origin,
            dimensions=(x, y, z), dtype=dtype)

# Time
nt = tmax // dt + 1
time_axis = TimeAxis(start=0.0, stop=tmax, step=dt)
print(time_axis)

# Velocity model
v = Function(name='v', grid=grid, space_order=space_order)
v.data[:] = 1.5

# Wavefield
u = TimeFunction(name='u', grid=grid, time_order=2, space_order=space_order)
t, x, y, z = u.dimensions

# Source
src = RickerSource(name='src', grid=v.grid, f0=fpeak, npoint=1,
                   time_range=time_axis)
src.coordinates.data[:, 0] = origin[0] + d * nx // 2
src.coordinates.data[:, 0] = d
src_term = src.inject(field=u.forward, expr=src * t.spacing**2 * v**2)

# Receivers
rec_coords = np.empty((nx, 3), dtype=dtype)
rec_coords[:, 0] = np.linspace(0.0, d * (nx - 1), nx)
rec_coords[:, 1] = origin[1] + d * (ny - 2) / 2
rec_coords[:, 2] = origin[1] + d * (nz - 2) / 2
rec = Receiver(name='rec', grid=grid, time_range=time_axis, coordinates=rec_coords)
rec_term = rec.interpolate(expr=u)

# Equation for the time update
eq_time_update = (t.spacing**2 * v**2) * \
    ((u.dx(x0=x+x.spacing/2)).dx(x0=x-x.spacing/2) +
     (u.dy(x0=y+y.spacing/2)).dy(x0=y-y.spacing/2) +
     (u.dz(x0=z+z.spacing/2)).dz(x0=z-z.spacing/2)) + \
    2 * u - u.backward

# Stencil
stencil = Eq(u.forward, eq_time_update)

# Substitute spacing terms to reduce flops
dt = time_axis.step
spacing_map = v.grid.spacing_map
spacing_map.update({t.spacing: dt})

# Operator
op = Operator([stencil] + src_term + rec_term, subs=spacing_map,
              name='MFE_Operator')

summary = op.apply()
