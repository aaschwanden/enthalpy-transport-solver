"""
Nonlinear 1-D enthalpy transport equation with Dirichlet conditions
at the surface and an inhomogeneous Neumann (flux) condition
at the base. The domain is the interval from a to b
-div(kappa(u)*nabla_grad(u) = 0,
u = u_surf at x=a, u=u_base at x=b.

Solution method: automatic, i.e., by a NonlinearVariationalProblem/Solver
(Newton method).

"""

from firedrake import *
import ufl
import sys
import numpy as np

tol = 1e-5


a = 0
b = 1_000
nx = 5_000
depth = -np.linspace(a, b, nx + 1, endpoint=True)
mesh = IntervalMesh(nx, a, b)
ele_order = 1
V = FunctionSpace(mesh, 'Lagrange', ele_order)



# Define variational problem
v  = TestFunction(V)
u  = TrialFunction(V)

theta  = 0.5      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

# better use udunits
spa = 31556926

c_i = 2009  # J kg-1 K-1
k_i = 2.1  # J m-1 K-1 s-1
rho_i = 910  # kg m-3
kappa = k_i/(c_i*rho_i) * spa

period = 1
t_a = 0.0
t_e = 1.0
dt = 1./12

u_0 = -13
delta_u_0 = 7.

t = t_a

# UFL equivalent:
x = SpatialCoordinate(V.mesh())
# u_surf = interpolate(u_0 + delta_u_0 * sin(2 * pi / period * t), V)
u_surf = Constant(0.0)

u_base = Constant(-13.)

bcs = [DirichletBC(V, u_surf, 1), DirichletBC(V, u_base, 2)]




u_sol = []

u_exact = interpolate(u_0 + delta_u_0 * exp(-x[0] * sqrt(2*pi/(2*kappa))) * sin(2 * pi / period * t -x[0] * sqrt(2 * pi/(2 * kappa))), V)


u_ = Function(V)  # most recently computed solution
u_prev = u_exact
t += dt


while t <= t_e:
    print('time: {}'.format(t))

    u_surf.assign(u_0 + delta_u_0 * sin(2 * pi / period * t))

    u_mid = (1.0-theta)*u_prev + theta*u

    F = (u - u_prev) * v * dx + dt * (inner(kappa * nabla_grad(u_mid), nabla_grad(v)) * dx)
    F  = action(F, u_)
    J = derivative(F, u_, u)

    # Compute solution
    problem = NonlinearVariationalProblem(F, u_, bcs, J)
    solver  = NonlinearVariationalSolver(problem)
    PROGRESS = 1
    set_log_level(PROGRESS)
    solver.solve()

    t += dt

    u_prev.assign(u_)
    u_sol.append(u_.vector().array())

import pylab as plt

period = 1
z = np.linspace(a, b, nx+1, endpoint=True)
A = np.sqrt(2*np.pi/(2*kappa))
T = u_0 + delta_u_0 * np.exp(-A * z) * np.sin(2 * np.pi / period * t - A * z)

fig=plt.figure()
ax = fig.add_subplot(111)

for k, sol in enumerate(u_sol):

    t= k*dt
    print(t)
    ax.plot(sol[::-1], -z[::-1])
    T = u_0 + delta_u_0*np.exp(-A*z)*np.sin(2*np.pi/period*t-A*z)
    ax.plot(T[::-1], -z[::-1])
ax.set_xlim(-20,0)
ax.set_ylim(-50, 0)


# Plot analytical solution for Colle Gnifetti
fig = plt.figure()
ax = fig.add_subplot(111)
for k in range(0, 12):
    t= k * dt
    T = u_0 + delta_u_0 * np.exp(-A * z) * np.sin(2 * np.pi / period * t - A * z)
    ax.plot( T[z<=20], z[z<=20], 'k')
ax.set_xlabel('temperature (degC)')
ax.set_ylabel('depth below surface (m)')
ax.invert_yaxis()
plt.savefig('colle-analytical.pdf')
plt.show()
