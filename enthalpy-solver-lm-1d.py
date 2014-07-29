"""
Nonlinear 1-D enthalpy transport equation with Dirichlet conditions
at the surface and an inhomogeneous Neumann (flux) condition
at the base. The domain is the interval from a to b.
Stationary Solver:
-div(q(E)*nabla_grad(E) + velocity*nabla_grad(e)) = f,
E = E_surf at x=a, dE/dn=g at x=b.
dEdt-div(q(E)*nabla_grad(E) + velocity*nabla_grad(e)) = f,

Solution method: automatic, i.e., by a NonlinearVariationalProblem/Solver
(Newton method).

ToDo: currently the pressure has to be a scalar in the EnthalpyConverter
"""

from dolfin import *
import sys
import numpy as np
import pylab as plt
from argparse import ArgumentParser

tol = 1E-14

class EnthalpyConverter(object):

    def __init__(self, config, *args, **kwargs):
        super(EnthalpyConverter, self).__init__(*args, **kwargs)

        self.config = config

    def getMeltingTemp(self, p):

        T_melting = self.config['T_melting']
        beta = self.config['beta']

        if hasattr(p, "__len__"):
            return T_melting - beta * np.array(p)
        else:
            return T_melting - beta * p

    def getEnth(self, T, omega, p):
        
        T_m = self.getMeltingTemp(p)

        c_i = self.config['c_i']
        T_0 = self.config['T_0']


        if hasattr(T, "__len__"):
            E = np.ones_like(T)
            T = np.array(T)
            omega = np.array(omega)
            idx = T < T_m
            E[idx] = c_i * (T[idx] - T_0)
            E[~idx] = self.getEnthalpyCTS(p) + omega[~idx] * L
        else:
            if (T < T_m):
                E = c_i * (T - T_0)
            else:
                E = self.getEnthalpyCTS(p) + omega * L

        return E
        
    def getEnthalpyCTS(self, p):

        c_i = self.config['c_i']
        T_0 = self.config['T_0']

        return c_i * (self.getMeltingTemp(p) - T_0)


    def getAbsTemp(self, E, p):

        E_s, E_l = self.getEnthalpyInterval(p)

        c_i = self.config['c_i']
        T_0 = self.config['T_0']

        if hasattr(E, "__len__"):
            T = np.ones_like(E)
            E = np.array(E)
            p = np.array(p)
            idx = E < E_s
            T[idx] =  (E[idx] / c_i) + T_0
            T[~idx] = self.getMeltingTemp(p)
        else:
            if (E < E_s):
                T = (E / c_i) + T_0
            else:
                T = self.getMeltingTemp(p)

        return T

    def getEnthalpyInterval(self, p):

        L = self.config['L']
        E_s = self.getEnthalpyCTS(p)
        E_l = E_s + L

        return E_s, E_l


    def getPATemp(self, E, p):
        
        T_melting = self.config['T_melting']
        T = self.getAbsTemp(E, p)
        T_pa = T - self.getMeltingTemp(p) + T_melting

        return T_pa


    def isTemperate(E, p):
        
        return (E>= self.getEnthalpyCTS(p))


    def getWaterFraction(self, E, p):
        
        E_s, E_l = self.getEnthalpyInterval(p)
        L = self.config['L']

        if (E >= E_l):
            omega = 1.0
        if (E <= E_s):
            omega = 0.
        else:
            omega = (E - E_s) / L

        return omega

    def isLiquified(E, p):

        E_s, E_l = self.getEnthalpyInterval(p)

        return (E >= E_l)

    def getEnthAtWaterFraction(self, omega, p):

        return self.getEnthalpyCTS(p) + omega * L

    def getPressureFromDepth(self, depth):
        
        p_air = self.config['p_air']
        rho_i = self.config['rho_i']
        g = self.config['g']

        if (depth > 0.):
            
            return p_air * rho_i * g * depth

        else:

            return p_air


class SteadyStateNonlinearSolver(object):
    '''
    Solves the (possibly) nonlinear steady-state problem:
    -div(kappa(E)*nabla_grad(E) + velocity*nabla_grad(e)) = f,
    E = E_surf at x=a, dE/dn=g at x=b.
    '''
    def __init__(self, kappa, velocity, f, g, bcs, *args, **kwargs):
        super(SteadyStateNonlinearSolver, self).__init__(*args, **kwargs)

        F = (inner(kappa*nabla_grad(E), nabla_grad(v))*dx 
             + inner(velocity*nabla_grad(E), nabla_grad(v))*dx 
             + f*v*dx - g*v*ds(2))

        E_ = Function(V)  # most recently computed solution
        F  = action(F, E_)
        J = derivative(F, E_, E)

        # Compute solution
        problem = NonlinearVariationalProblem(F, E_, bcs, J)
        solver  = NonlinearVariationalSolver(problem)
        prm = solver.parameters
        info(prm, True)
        prm['newton_solver']['absolute_tolerance'] = 1E-8
        prm['newton_solver']['relative_tolerance'] = 1E-7
        prm['newton_solver']['maximum_iterations'] = 25
        prm['newton_solver']['relaxation_parameter'] = 1.0
        PROGRESS = 16
        set_log_level(PROGRESS)
        solver.solve()

        self.E_ = E_
        self.F = F
        self.J = J
        self.problem = problem
        self.solver = solver

    def run(self):
        self.solver.solve()


class DirichletBCTransientNonlinearSolver(object):

    def __init__(self, kappa, velocity, f, g, bcs, *args, **kwargs):
        super(DirichletBCTransientNonlinearSolver, self).__init__()

        # get time control parameters
        time_control = kwargs['time_control']

        t_a = time_control['t_a']
        t_e = time_control['t_e']
        dt = time_control['dt']
        theta = time_control['theta']

        E_sol = []

        t = t_a
        print('time: {} (start)'.format(t))

        # get initial condition
        E_init = kwargs['E_init']
        E_init.t = t

        E_surf, E_base = bcs
        E_surf.t = t_a

        E_ = Function(V)  # most recently computed solution

        c_i = EC.config['c_i']
        k_i = EC.config['k_i']
        kappa_0 = k_i / c_i / rho_i * spa 
        
        # We use the exact solution of the linear diffusion problem
        # as an initial condition at t=t_a
        E0 = interpolate(E_init, V)

        E_prev = E0
        E_sol.append(E0.vector().array())
        t += dt

        while t <= t_e:
            print('time: {}'.format(t))
            
            # Update surface boundary condition
            E_surf.t = t
            bcs = [DirichletBC(V, E_surf, boundary_parts, 1), DirichletBC(V, Constant(E_base), boundary_parts, 2)]

            # E_(n+theta)
            E_mid = (1.0-theta)*E_prev + theta*E

            F = ((E-E_prev)*v*dx + dt*(inner(kappa*nabla_grad(E_mid), nabla_grad(v))*dx 
                + inner(velocity*nabla_grad(E_mid), nabla_grad(v))*dx 
                + f*v*dx))

            F  = action(F, E_)
            J = derivative(F, E_, E)

            # Compute solution
            problem = NonlinearVariationalProblem(F, E_, bcs, J)
            solver  = NonlinearVariationalSolver(problem)
            prm = solver.parameters
            prm['newton_solver']['absolute_tolerance'] = 1E-8
            prm['newton_solver']['relative_tolerance'] = 1E-7
            prm['newton_solver']['maximum_iterations'] = 25
            prm['newton_solver']['relaxation_parameter'] = 1.0
            PROGRESS = 1
            set_log_level(PROGRESS)
            solver.solve()

            t += dt

            E_prev.assign(E_)
            E_sol.append(E_.vector().array())
        print('Time stepping done')

        # Fixme: What should we return??
        self.E_ = E_
        self.E_sol = E_sol

class TransientNonlinearSolver(object):

    def __init__(self, kappa, velocity, f, g, bcs, *args, **kwargs):
        super(TransientNonlinearSolver, self).__init__()

        # get time control parameters
        time_control = kwargs['time_control']

        t_a = time_control['t_a']
        t_e = time_control['t_e']
        dt = time_control['dt']
        theta = time_control['theta']

        E_sol = []

        t = t_a
        print('time: {} (start)'.format(t))

        # get initial condition
        E_init = kwargs['E_init']
        E_init.t = t

        E_surf = bcs[0]
        E_surf.t = t_a

        E_ = Function(V)  # most recently computed solution

        c_i = EC.config['c_i']
        k_i = EC.config['k_i']
        kappa_0 = k_i / c_i / rho_i * spa 
        
        # We use the exact solution of the linear diffusion problem
        # as an initial condition at t=t_a
        E0 = interpolate(E_init, V)

        E_prev = E0
        E_sol.append(E0.vector().array())
        t += dt

        while t <= t_e:
            print('time: {}'.format(t))
            
            # Update surface boundary condition
            E_surf.t = t
            bcs = [DirichletBC(V, E_surf, boundary_parts, 1)]

            # E_(n+theta)
            E_mid = (1.0-theta)*E_prev + theta*E

            F = ((E-E_prev)*v*dx + dt*(inner(kappa*nabla_grad(E_mid), nabla_grad(v))*dx 
                + inner(velocity*nabla_grad(E_mid), nabla_grad(v))*dx 
                                       + f*v*dx - g*v*ds(2)))

            F  = action(F, E_)
            J = derivative(F, E_, E)

            # Compute solution
            problem = NonlinearVariationalProblem(F, E_, bcs, J)
            solver  = NonlinearVariationalSolver(problem)
            prm = solver.parameters
            prm['newton_solver']['absolute_tolerance'] = 1E-8
            prm['newton_solver']['relative_tolerance'] = 1E-7
            prm['newton_solver']['maximum_iterations'] = 25
            prm['newton_solver']['relaxation_parameter'] = 1.0
            PROGRESS = 1
            set_log_level(PROGRESS)
            solver.solve()

            t += dt

            E_prev.assign(E_)
            E_sol.append(E_.vector().array())
        print('Time stepping done')

        # Fixme: What should we return??
        self.E_ = E_
        self.E_sol = E_sol

class LMTransientNonlinearSolver(object):

    def __init__(self, kappa, velocity, f, g, bcs, *args, **kwargs):
        super(LMTransientNonlinearSolver, self).__init__()

        # get time control parameters
        time_control = kwargs['time_control']

        t_a = time_control['t_a']
        t_e = time_control['t_e']
        dt = time_control['dt']
        theta = time_control['theta']

        E_sol = []

        t = t_a
        print('time: {} (start)'.format(t))

        # get initial condition
        E_init = kwargs['E_init']
        E_init.t = t

        E_surf = bcs[0]
        E_surf.t = t_a

        E_ = Function(W)  # most recently computed solution

        c_i = EC.config['c_i']
        k_i = EC.config['k_i']
        kappa_0 = k_i / c_i / rho_i * spa 
        
        # We use the exact solution of the linear diffusion problem
        # as an initial condition at t=t_a
        E0 = interpolate(E_init, V)

        E_prev = E0
        E_sol.append(E0.vector().array())
        t += dt

        while t <= t_e:
            print('time: {}'.format(t))
            
            # Update surface boundary condition
            E_surf.t = t
            bcs = [DirichletBC(V, E_surf, boundary_parts, 1)]

            # E_(n+theta)
            E_mid = (1.0-theta)*E_prev + theta*E

            F = ((E-E_prev)*v*dx + dt*(inner(kappa*nabla_grad(E_mid), nabla_grad(v))*dx 
                + inner(velocity*nabla_grad(E_mid), nabla_grad(v))*dx 
                + f*v*dx + (lm*v + (E-1000.)*v_lm*ds(2) - g*v*ds(2)))

            F  = action(F, E_)
            J = derivative(F, E_, E)

            # Compute solution
            problem = NonlinearVariationalProblem(F, E_, bcs, J)
            solver  = NonlinearVariationalSolver(problem)
            prm = solver.parameters
            prm['newton_solver']['absolute_tolerance'] = 1E-8
            prm['newton_solver']['relative_tolerance'] = 1E-7
            prm['newton_solver']['maximum_iterations'] = 25
            prm['newton_solver']['relaxation_parameter'] = 1.0
            PROGRESS = 1
            set_log_level(PROGRESS)
            solver.solve()

            t += dt

            E_prev.assign(E_)
            E_sol.append(E_.vector().array())
        print('Time stepping done')

        # Fixme: What should we return??
        self.E_ = E_
        self.E_sol = E_sol

class Verification(object):

    def __init__(self, EC, *args, **kwargs):
        super(Verification, self).__init__(*args, **kwargs)
        self.EC = EC

    def transient_diffusion(self):
        '''
        Verify the constant-coefficient transient diffusion problem

        dEdt-div(kappa*nabla_grad(E) = 0
        '''

        EC = self.EC
        c_i = EC.config['c_i']
        k_i = EC.config['k_i']
        rho_i =  EC.config['rho_i']
        p_air = EC.config['p_air']
        kappa= k_i / c_i / rho_i * spa 

        f = 0  # no production
        velocity = 0  # zero velocity

        T_base = 260.
        T_surf = 260.

        E_0 = EC.getEnth(T_surf, 0., p_air)
        T_amplitude = 5.  # K (needs to be float!!)
        period = 1
        T_plus  = T_surf+T_amplitude/2
        T_minus = T_surf-T_amplitude/2
        delta_E_0 = (EC.getEnth(T_plus, 0., p_air) - EC.getEnth(T_minus, 0., p_air))
        dt = 1./12
        t_a = 0  # start at year zero
        t_e = 1  # end at year one
        theta = 0.5
        time_control = dict(t_a=t_a, t_e=t_e, dt=dt, theta=theta)

        # Surface boundary condition
        E_surf = Expression('E_0 + delta_E_0*sin(2*pi/period*t)', E_0=E_0, delta_E_0=delta_E_0, period=period, t=t_a)
        # Basal boundary condition
        E_base = EC.getEnth(T_base, 0., p_air)

        lm_constraint = DirichletBC(W.sub(1), 0.0, "abs(x[1])>2.0*DOLFIN_EPS")

        # Combine boundary conditions
        bcs = [E_surf, E_base, lm_constraint]

        # Define exact solution used as initial condition:
        E_exact = Expression('E_0 + delta_E_0*exp(-x[0]*sqrt((2*pi)/(2*kappa)))*sin(2*pi*t-x[0]*sqrt((2*pi)/(2*kappa)))', 
                             E_0=E_0, delta_E_0=delta_E_0, kappa=kappa, t=t_a)




        transient_problem = DirichletBCTransientNonlinearSolver(Constant(kappa), Constant(velocity), f, g, bcs, E_init=E_exact, time_control=time_control)
        E_sol = transient_problem.E_sol

        # extract solution at t=t_e
        E_sol_te = transient_problem.E_sol[-1]
        E_exact = Expression('E_0 + delta_E_0*exp(-x[0]*sqrt((2*pi)/(2*kappa)))*sin(2*pi*t-x[0]*sqrt((2*pi)/(2*kappa)))', 
                             E_0=E_0, delta_E_0=delta_E_0, kappa=kappa, t=t_e)
        # exact solution at time t_e
        E_exact.t = t_e
        E_e = interpolate(E_exact, V)
        diff = np.abs(EC.getAbsTemp(E_e.vector().array(), p_air) - EC.getAbsTemp(E_sol_te, p_air)).max()
        print('Max error: {:2.3f} J kg-1'.format(diff))
        z = np.linspace(a, b, nx+1, endpoint=True)
        A = np.sqrt(2*np.pi/(2*kappa))
        period = 1

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for k, sol in enumerate(E_sol):
            t = dt*k
            ax.plot(EC.getAbsTemp(sol, p_air)[z<=20],'-', color='b')
            E_e = E_0 + delta_E_0*np.exp(-A*z)*np.sin(2*np.pi/period*t-A*z)
            ax.plot(EC.getAbsTemp(E_e, p_air)[z<20], ':', color='k')
        plt.legend(['approx','exact'])


    def steady_state_diffusion(self):
        '''
        Verify the constant-coefficient steady-state diffusion problem

        -div(kappa*nabla_grad(E) = 0
        '''

        EC = self.EC
        c_i = EC.config['c_i']
        k_i = EC.config['k_i']
        p_air = EC.config['p_air']
        q_geo = EC.config['q_geo']  # heat flux at the base

        kappa= k_i / c_i

        f = 0  # no production
        velocity = 0  # zero velocity

        T_surf = 263.  #
        E_surf = EC.getEnth(T_surf, 0., p_air)

        bcs = [DirichletBC(V, E_surf, boundary_parts, 1)]

        steady_state = SteadyStateNonlinearSolver(kappa, velocity, f, g, bcs)
        steady_state.run()
        E_ = steady_state.E_
        E_exact = Expression('E_surf + q_geo/k_i*c_i*x[0]', E_surf=E_surf, q_geo=q_geo, k_i=k_i, c_i=c_i)
        E_e = interpolate(E_exact, V)
        diff = np.abs(E_e.vector().array() - E_.vector().array()).max()
        print('Max error: {:2.3f} J kg-1'.format(diff))


    def run(self):
        self.steady_state_diffusion()
        self.transient_diffusion()

# Set up the option parser
parser = ArgumentParser()
parser.description = "1-D enthalpy transport solver."
parser.add_argument("--verify", dest="do_verification", action='store_true',
                    help='''Run verification tests. Default=False.''', default=False)
options = parser.parse_args()

do_verification = options.do_verification

a = 0
b = 1000
nx = 1000
mesh = IntervalMesh(nx, a, b)
ele_order = 1
V = FunctionSpace(mesh, 'Lagrange', ele_order)
W = V*V

c_i = 2009  # J kg-1 K-1
c_w = 4170  # J kg-1 K-1
k_i = 2.1  # J m-1 K-1 s-1
L = 3.34e5  # J kg-1
rho_i = 910  # kg m-3
T_melting = 273.15  # K
T_0 = 223.15  # K
g = 9.81  # m s-2
beta = 7.9e-8  # K Pa-1
p_air = 1e5  # Pa
q_geo = 0.042  # W m-2
kappa_0 = 1e-5


config = dict(c_i=c_i, c_w=c_w, k_i=k_i, L=L, rho_i=rho_i, T_melting=T_melting, 
              T_0=T_0, g=g, beta=beta, p_air=p_air, q_geo=q_geo)

EC = EnthalpyConverter(config)



# Define boundary conditions

class SurfaceBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]) < tol

class LowerBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]-b) < tol

boundary_parts = FacetFunction("size_t", mesh, 1)
boundary_parts.set_all(0)

Gamma_d = SurfaceBoundary()
Gamma_d.mark(boundary_parts, 1)

Gamma_g = LowerBoundary()
Gamma_g.mark(boundary_parts, 2)


# Define variational problem
v, v_lm  = TestFunction(W)
E, lm  = TrialFunction(W)
f = Constant(0.)
ds = ds[boundary_parts]
g = Constant(config['q_geo'])
spa = 24 * 3600 *365

c_i = EC.config['c_i']
k_i = EC.config['k_i']
rho_i =  EC.config['rho_i']
p_air = EC.config['p_air']
kappa= k_i / c_i / rho_i * spa 

f = 0  # no production
velocity = 0  # zero velocity

T_base = 260.
T_surf = 250.

E_0 = EC.getEnth(T_surf, 0., p_air)
T_amplitude = 5.  # K (needs to be float!!)
period = 1
T_plus  = T_surf+T_amplitude/2
T_minus = T_surf-T_amplitude/2
delta_E_0 = (EC.getEnth(T_plus, 0., p_air) - EC.getEnth(T_minus, 0., p_air))
dt = 5
t_a = 0  # start at year zero
t_e = 10  # end at year one
theta  = 0.5      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

time_control = dict(t_a=t_a, t_e=t_e, dt=dt, theta=theta)
E_surf = Expression('E_0 + delta_E_0*sin(2*pi/period*t)', E_0=E_0, delta_E_0=delta_E_0, period=period, t=t_a)
#E_base = EC.getEnth(T_base, 0., p_air)
bcs = [E_surf]

acab = 1
velocity = Expression('acab-acab/(b-a)*x[0]', acab=acab, a=a, b=b)

if do_verification:
    verify = Verification(EC)
    verify.run()
else:

    ## This needs to be turned into a FEniCS expression:
    def kappa_cold(E):

        return k_i / (146.3 + 7.253 * E / c_i) / rho_i * spa

    # Define exact solution used as initial condition:
    E_exact = Expression('E_surf + q_geo/k_i*c_i*x[0]', E_surf=E_surf, q_geo=q_geo, k_i=k_i, c_i=c_i)


    transient_problem = LMTransientNonlinearSolver(Constant(kappa), velocity, f, g, bcs, E_init=E_exact, time_control=time_control)
    E_sol = transient_problem.E_sol

    z = np.linspace(a, b, nx+1, endpoint=True)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # for sol in E_sol:
    #     depth = -z[::-1]
    #     T = EC.getAbsTemp(sol[::-1], p_air)
    #     ax.plot(T, depth)
    # # ax.set_ylim(-50, 0)


    fig = plt.figure()
    ax = fig.add_subplot(111)
    depth = -z[::-1]
    T = EC.getAbsTemp(E_sol[0][::-1], p_air)
    ax.plot(T, depth)
    T = EC.getAbsTemp(E_sol[-1][::-1], p_air)
    ax.plot(T, depth)
    # ax.set_ylim(-50, 0)
