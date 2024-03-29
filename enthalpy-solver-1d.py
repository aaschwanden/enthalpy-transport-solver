"""
Nonlinear 1-D enthalpy transport equation with Dirichlet conditions
at the surface and an inhomogeneous Neumann (flux) condition
at the base. The domain is the interval from a to b.
Stationary Solver:
-div(q(E)*nabla_grad(E) + velocity*nabla_grad(E)) = f,
E = E_surf at x=a, dE/dn=g at x=b.
Transient Solver:
dEdt-div(q(E)*nabla_grad(E) + velocity*nabla_grad(E)) = f,

Solution method: automatic, i.e., by a NonlinearVariationalProblem/Solver
(Newton method).

ToDo: currently the pressure has to be a scalar in the EnthalpyConverter
"""

from firedrake import *
import sys
import numpy as np
import pylab as plt
from argparse import ArgumentParser

set_log_level(ERROR)
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
        T_pa = T - self.getMeltingTemp(p)

        return T_pa


    def isTemperate(E, p):
        
        return (E >= self.getEnthalpyCTS(p))


    def getWaterFraction(self, E, p):
        
        E_s, E_l = self.getEnthalpyInterval(p)
        L = self.config['L']

        if hasattr(E, "__len__"):
            omega = np.ones_like(E)
            E = np.array(E)
            p = np.array(p)
            idx = E < E_s
            omega[idx] = 0.
            omega[~idx] = (E[~idx] - E_s) / L
            idx = E >= E_l
            omega[idx] = 1.
        else:
            if (E >= E_l):
                omega = 1.
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
            
            return p_air + rho_i * g * depth

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

        # necessary quantities for streamline upwinding :
        h      = 2 * CellSize(mesh)

        # skewed test function :
        psihat = psi + h / 2 * sign(velocity) * psi.dx(0)

        F =  (inner(kappa * nabla_grad(E), nabla_grad(psi)) * dx 
             + velocity * E.dx(0) * psihat * dx 
             + f * psihat * dx - g * psihat * ds(2)
        )

        E_ = Function(V)  # most recently computed solution
        F  = action(F, E_)
        J = derivative(F, E_, E)

        # Compute solution
        problem = NonlinearVariationalProblem(F, E_, bcs, J)
        solver  = NonlinearVariationalSolver(problem)
        prm = solver.parameters
        info(prm, True)
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

        print('--------------------------------------------------------')
        print('Running Transient Nonlinear Solver with Dirichlet BCs')
        print('--------------------------------------------------------\n')

        t = t_a
        print('time: {} (start)'.format(t))

        E_sol = []

        # get initial condition
        E_init = kwargs['E_init']
        E_init.t = t

        E_surf, E_base = bcs
        E_surf.t = t_a

        E_ = Function(V)  # most recently computed solution

        
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

            # E_ (n + theta)
            E_mid = (1.0 - theta) * E_prev + theta * E

            # necessary quantities for streamline upwinding :
            h      = 2 * CellSize(mesh)

            # skewed test function :
            psihat = psi + h / 2 * sign(velocity) * psi.dx(0)

            F = ((E - E_prev) * psihat * dx + dt * (inner(kappa * nabla_grad(E_mid), nabla_grad(psi)) * dx 
                + velocity * E.dx(0) * psihat * dx 
                + f * psihat * dx))

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

        print('--------------------------------------------------------')
        print('Running Transient Nonlinear Solver')
        print('--------------------------------------------------------')

        t = t_a
        print('time: {} (start)'.format(t))

        E_sol = []

        # get initial condition
        E_init = kwargs['E_init']
        E_init.t = t

        E_surf = bcs[0]
        E_surf.t = t_a

        E_ = Function(V)  # most recently computed solution
        
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
            E_mid = (1.0 - theta) * E_prev + theta * E

            # necessary quantities for streamline upwinding :
            h      = 2 * CellSize(mesh)

            # skewed test function :
            psihat = psi + h / 2 * sign(velocity) * psi.dx(0)

            # kappa should be kappa_mid = kappa(E_mid)
            F = ((E - E_prev) * psihat * dx + dt * (inner(kappa * nabla_grad(E_mid), nabla_grad(psi)) * dx 
                + velocity *E.dx(0) * psihat * dx 
                + f * psihat * dx - g * psihat * ds(2)))

            F = action(F, E_)
            J = derivative(F, E_, E)

            # Compute solution
            problem = NonlinearVariationalProblem(F, E_, bcs, J)
            solver  = NonlinearVariationalSolver(problem)
            prm = solver.parameters
            prm['newton_solver']['absolute_tolerance'] = 1E-8
            prm['newton_solver']['relative_tolerance'] = 1E-7
            prm['newton_solver']['maximum_iterations'] = 25
            prm['newton_solver']['relaxation_parameter'] = 1.0
            solver.solve()

            # Form representing the basal melt rate
            term  = q_geo - (-rho_i * kappa * E_.dx(0))
            q_friction = 0
            Mb    = (q_friction + term) / (L * rho_i)

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

        dEdt - div(kappa*nabla_grad(E) = 0
        '''

        EC = self.EC
        c_i = EC.config['c_i']
        k_i = EC.config['k_i']
        rho_i =  EC.config['rho_i']
        p_air = EC.config['p_air']

        kappa = k_i / (c_i * rho_i)

        f = Constant(0.)
        g = Constant(q_geo / rho_i)
        velocity = 0  # zero velocity

        T_base = 260.
        T_surf = 260.

        E_0 = EC.getEnth(T_surf, 0., p_air)
        T_amplitude = 5.  # K (needs to be float!!)
        period = 1 
        T_plus  = T_surf + T_amplitude / 2
        T_minus = T_surf - T_amplitude/ 2
        delta_E_0 = (EC.getEnth(T_plus, 0., p_air) - EC.getEnth(T_minus, 0., p_air))
        dt = 1./12 
        t_a = 0  # start at year zero
        t_e = 1 # end at year one

        thetas = [0.5, 1.0]
        E_sols = []
        for theta in thetas:
            time_control = dict(t_a=t_a, t_e=t_e, dt=dt, theta=theta)

            t = 0
            # Surface boundary condition
            E_surf = interpolate(E_0 + delta_E_0 * sin(2 * pi / period * t), V)
            # Basal boundary condition
            E_base = EC.getEnth(T_base, 0., p_air)
            # Combine boundary conditions
            bcs = [E_surf, E_base]

            # Define exact solution used as initial condition:
            E_exact = interpolate(E_0 + delta_E_0 * exp(-x[0] * sqrt((2 * pi)/(2 * kappa))) * sin(2 * pi / period * t - x[0] * sqrt((2 * pi)/(2 * kappa))), V)


            transient_problem = DirichletBCTransientNonlinearSolver(Constant(kappa), Constant(velocity), f, g, bcs, E_init=E_exact, time_control=time_control)
            E_sol = transient_problem.E_sol

            E_sols.append(E_sol)

        E_exact = interpolate(E_0 + delta_E_0*exp(-x[0]*sqrt((2*pi)/(2*kappa)))*sin(2*pi/period*t-x[0]*sqrt((2*pi)/(2*kappa))), V)
        E_exact.t = t_e

        E_e = interpolate(E_exact, V)
        # Max difference between exact solution and Crank-Nicolson
        E_sol_te_cn = E_sols[0][-1]
        diff_cn = np.abs(EC.getAbsTemp(E_e.vector().array(), p_air) - EC.getAbsTemp(E_sol_te_cn, p_air)).max()
        # Max difference between exact solution and backward Euler
        E_sol_te_be = E_sols[1][-1]
        diff_bw = np.abs(EC.getAbsTemp(E_e.vector().array(), p_air) - EC.getAbsTemp(E_sol_te_be, p_air)).max()
        print('\n--------------------------------------------------------')
        print('Verification: transient diffusion\n')
        print('Max error Crank-Nicolson: {:2.3f} K'.format(diff_cn))
        print('Max error Backward-Euler: {:2.3f} K'.format(diff_bw))
        print('--------------------------------------------------------\n')
        z = np.linspace(a, b, nx+1, endpoint=True)
        A = np.sqrt(2 * np.pi/ (2 * kappa))
        period = 1

        fig = plt.figure()
        ax = fig.add_subplot(111)

        for k in range(len(E_sols[0])):
            t = dt * k
            depth = z[z<=20]
            ax.plot(EC.getAbsTemp(E_sols[0][k], p_air)[z<=20], depth, '-', color='b')
            ax.plot(EC.getAbsTemp(E_sols[1][k], p_air)[z<=20], depth, '-', color='r')
            E_e = E_0 + delta_E_0*np.exp(-A*z)*np.sin(2*np.pi/period*t-A*z)
            ax.plot(EC.getAbsTemp(E_e, p_air)[z<=20], depth, ':', color='k')

        ax.set_xlabel('temperature (K)')
        ax.set_ylabel('depth below surface (m)')
        ax.invert_yaxis()
        plt.title('Verification: Transient Diffusion')
        plt.legend(['Crank-Nicolson', 'Backward-Euler', 'exact solution'])


    def steady_state_diffusion(self):
        '''
        Verify the constant-coefficient steady-state diffusion problem

        -div(kappa*nabla_grad(E)) = 0
        '''

        EC = self.EC
        c_i = EC.config['c_i']
        k_i = EC.config['k_i']
        rho_i = EC.config['rho_i']
        p_air = EC.config['p_air']
        q_geo = EC.config['q_geo']  # heat flux at the base
        f = Constant(0.)
        g = Constant(q_geo/rho_i)
        
        kappa = c_i / k_i / rho_i

        f = 0  # no production
        velocity = 0  # zero velocity

        T_surf = 263.  #
        E_surf = EC.getEnth(T_surf, 0., p_air)

        bcs = [DirichletBC(V, E_surf, boundary_parts, 1)]

        steady_state = SteadyStateNonlinearSolver(kappa, velocity, f, g, bcs)
        steady_state.run()
        E_ = steady_state.E_
        E_exact = Expression('E_surf + q_geo/k_i*c_i*x[0]', E_surf=E_surf, q_geo=q_geo, k_i=k_i, c_i=c_i, degree=1)
        E_e = interpolate(E_exact, V)
        diff = np.abs(E_e.vector().array() - E_.vector().array()).max()
        print('\n--------------------------------------------------------')
        print('Verification: steady-state diffusion\n')
        print('Max error: {:2.3f} J kg-1'.format(diff))
        print('--------------------------------------------------------\n')


    def steady_state_advection_diffusion(self):
        '''
        Verify the constant-coefficient steady-state advection-diffusion problem

        -div(kappa*nabla_grad(E)) - velocity*nabla_grad(E) = 0
        '''
        
        from scipy.special import erf
        EC = self.EC
        c_i = EC.config['c_i']
        k_i = EC.config['k_i']
        rho_i = EC.config['rho_i']
        p_air = EC.config['p_air']
        q_geo = EC.config['q_geo']  # heat flux at the base
        f = Constant(0.)
        g = Constant(q_geo/rho_i)

        kappa = k_i / c_i / rho_i

        f = 0  # no production
        acab = 1 # m/a
        velocity = interpolate(acab-acab/(b-a)*x[0], V)

        T_surf = 253.  #
        E_surf = EC.getEnth(T_surf, 0., p_air)

        bcs = [DirichletBC(V, E_surf, 1)]

        steady_state = SteadyStateNonlinearSolver(kappa, velocity, f, g, bcs)
        steady_state.run()
        E_ = steady_state.E_
        E_sol = E_.vector().array() 
        T_computed = EC.getAbsTemp(E_sol, p_air)       
        z = np.linspace(a, b, nx+1, endpoint=True)
        l = np.sqrt(2*kappa*(b-a)/acab)
        dTdz = -q_geo / k_i 
        T_exact = T_surf + np.sqrt(np.pi)/2*l*dTdz*(erf((b-z)/l)-erf((b-a)/l))

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(T_computed, z, '-', color='b')
        ax.plot(T_exact, z, ':', color='k')
        ax.set_xlabel('temperature (K)')
        ax.set_ylabel('depth below surface (m)')
        ax.invert_yaxis()
        plt.title('Verification: Steady-State Advection-Diffusion')
        plt.legend(['computed', 'exact solution'])

        # E_exact = Expression('E_surf + np.sqrt(np.pi)/2*l*c_i*dTdz*erf((b-x[0])/l)-erf((b-a)/l)', E_surf=E_surf, 
        #                      l=l, dTdz=dTdz, c_i=c_i, a=a, b=b)
        # E_e = interpolate(E_exact, V)
        diff = np.abs(T_exact - T_computed).max()
        print('\n--------------------------------------------------------')
        print('Verification: steady-state advection-diffusion\n')
        print('Max error: {:2.3f} K'.format(diff))
        print('--------------------------------------------------------\n')

    def run(self):
        print('Running transient diffusion verification')
        self.transient_diffusion()
        print('Running steady-sate diffusion verification')
        self.steady_state_diffusion()
        print('Running steady-state advection-diffusion verification')
        self.steady_state_advection_diffusion()


# Set up the option parser
parser = ArgumentParser()
parser.description = "1-D enthalpy transport solver."
parser.add_argument("--verify", dest="do_verification", action='store_true',
                    help='''Run verification tests. Default=False.''', default=False)
options = parser.parse_args()

do_verification = options.do_verification


# Set up geometry and mesh
a = 0
b = 1000
nx = 1000
mesh = IntervalMesh(nx, a, b)
ele_order = 1
# Define function space
V = FunctionSpace(mesh, 'Lagrange', ele_order)

# Do convertion from s to yr here
secpera = 31556925.9747
c_i = 2009  # J kg-1 K-1
c_w = 4170  # J kg-1 K-1
k_i = 2.1 * secpera  # J m-1 K-1 yr-1
L = 3.34e5  # J kg-1
rho_i = 910  # kg m-3
T_melting = 273.15  # K
T_0 = 223.15  # K
g_acc = 9.81 * secpera**2 # m yr-2
beta = 7.9e-8  # K Pa-1
p_air = 1e5  # Pa
q_geo = 0.042 * secpera # J yr-1 m-2
kappa_0 = 1e-6 * secpera


config = dict(c_i=c_i, c_w=c_w, k_i=k_i, L=L, rho_i=rho_i, T_melting=T_melting, 
              T_0=T_0, g_acc=g_acc, beta=beta, p_air=p_air, q_geo=q_geo, secpera=secpera,kappa_0=kappa_0)

EC = EnthalpyConverter(config)


# Define boundary conditions

# class SurfaceBoundary(SubDomain):
#     def inside(self, x, on_boundary):
#         return on_boundary and abs(x[0]) < tol

# class LowerBoundary(SubDomain):
#     def inside(self, x, on_boundary):
#         return on_boundary and abs(x[0] - b) < tol

# Define variational problem
psi  = TestFunction(V)
E  = TrialFunction(V)
E_mid = E
f = Constant(0.)


c_i = EC.config['c_i']
k_i = EC.config['k_i']
rho_i =  EC.config['rho_i']
p_air = EC.config['p_air']
g_acc = EC.config['g_acc']
kappa_cold = k_i / c_i / rho_i
kappa_temperate = EC.config['kappa_0']

x = SpatialCoordinate(V.mesh())

p = interpolate(p_air + rho_i * g_acc * x[0], V)
T_pa = interpolate(T_melting - beta * p, V)
E_s = interpolate(c_i * (T_pa - T_0), V)
isTemperate = conditional(ge(E, E_s), 1, 0)


def K_i(T):
    '''
    Temperature-dependent cold-ice thermal conductivity

    Equation 4.36 in Greve & Blatter (2009)


    '''
    return 9.828 * np.exp(-0.0057 * T)

def C_i(T):
    '''
    Temperature-dependent cold-ice heat capacity

    Equation 4.39 in Greve & Blatter (2009)
    '''

    return (146.3 + 7.253 * T)

def Kappa_i(T):
    '''
    Temperature-dependent cold-ice diffusivity
    '''
    return K_i(T) / C_i(T) / rho_i


def kappa(E):
    condition = lt(E, E_s)
    return conditional(condition, kappa_cold, kappa_temperate)

 
    
if do_verification:
    verify = Verification(EC)
    verify.run()
else:

    # This is a little example with a temperate base

    f = Constant(0.)
    g = Constant(q_geo / rho_i)

    T_surf = 270
    E_0 = EC.getEnth(T_surf, 0., p_air)

    dt = 100
    t_a = 0  # start at year zero
    t_e = 5000  # end at year one
    theta  = 0.5      # time stepping family, e.g. theta=1 -> backward Euler, theta=0.5 -> Crank-Nicolson

    time_control = dict(t_a=t_a, t_e=t_e, dt=dt, theta=theta)
    E_surf = Constant(E_0)

    bcs = [E_surf]

    Mb = 0
    acab = 1
    velocity = interpolate((acab - (acab / (b-a)) * x[0] + Mb), V)
    
    E_init = interpolate(E_0, V)
    transient_problem = TransientNonlinearSolver(kappa(E_mid), velocity, f, g, bcs, E_init=E_init, time_control=time_control)
    E_sol = transient_problem.E_sol

    z = np.linspace(a, b, nx+1, endpoint=True)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    depth = z
    T = EC.getPATemp(E_sol[-1], p_air)
    lT, = ax.plot(T, depth, color='#08519c', label='pressure-adjusted temperature')
    omega = EC.getWaterFraction(E_sol[-1], p_air)
    ax_o = ax.twiny()
    lomega, = ax_o.plot(omega, depth, ':', color='#a50f15', label='water content')
    ax.set_xlabel(u'temperature (\u00B0C)')
    ax_o.set_xlabel('liquid water fraction (-)')
    ax.set_ylabel('depth below surface (m)')
    ax.invert_yaxis()
    # plt.legend(labels)

plt.show()
