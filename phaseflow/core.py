"""This module contains the core functionalty of Phaseflow."""
import fenics
import h5py
import numpy
import phaseflow.helpers

TIME_EPS = 1.e-8

pressure_degree = 1

temperature_degree = 1

""" The equations are scaled with unit Reynolds Number
per Equation 8 from danaila2014newton, i.e.

    v_ref = nu_liquid/H => t_ref = nu_liquid/H^2 => Re = 1.
"""
reynolds_number = 1.

MAX_TIME_STEPS = 1000000000000

def make_mixed_fe(cell):
    """ Define the mixed finite element.
    MixedFunctionSpace used to be available but is now deprecated. 
    To create the mixed space, I'm using the approach from https://fenicsproject.org/qa/11983/mixedfunctionspace-in-2016-2-0
    """
    velocity_degree = pressure_degree + 1
    
    velocity_element = fenics.VectorElement("P", cell, velocity_degree)
    
    pressure_element = fenics.FiniteElement("P", cell, pressure_degree)

    temperature_element = fenics.FiniteElement("P", cell, temperature_degree)

    mixed_element = fenics.MixedElement([velocity_element, pressure_element, temperature_element])
    
    return mixed_element

    
def write_solution(solution_file, w_m, time):
    """Write the solution to disk."""

    phaseflow.helpers.print_once("Writing solution to HDF5+XDMF")
    
    velocity, pressure, temperature = w_m.leaf_node().split()
    
    velocity.rename("u", "velocity")
    
    pressure.rename("p", "pressure")
    
    temperature.rename("T", "temperature")
    
    for i, var in enumerate([velocity, pressure, temperature]):
    
        solution_file.write(var, time)
        

def steady(W, w, w_n, steady_relative_tolerance):
    """Check if solution has reached an approximately steady state."""
    steady = False
    
    time_residual = fenics.Function(W)
    
    time_residual.assign(w - w_n)
    
    unsteadiness = fenics.norm(time_residual, "L2")/fenics.norm(w_n, "L2")
    
    phaseflow.helpers.print_once(
        "Unsteadiness (L2 norm of relative time residual), || w_{n+1} || / || w_n || = "+str(unsteadiness))

    if (unsteadiness < steady_relative_tolerance):
        
        steady = True
    
    return steady
  
  
def run(output_dir = "output/wang2010_natural_convection_air",
        rayleigh_number = 1.e6,
        prandtl_number = 0.71,
        stefan_number = 0.045,
        liquid_heat_capacity = 1.,
        solid_heat_capacity = 1.,
        liquid_thermal_conductivity = 1.,
        solid_thermal_conductivity = 1.,
        liquid_viscosity = 1.,
        solid_viscosity = 1.e8,
        gravity = (0., -1.),
        m_B = None,
        dm_B = None,
        penalty_parameter = 1.e-7,
        regularization_central_temperature= -1.e12,
        regularization_smoothing_factor = 0.005,
        mesh = fenics.UnitSquareMesh(fenics.dolfin.mpi_comm_world(), 20, 20, "crossed"),
        initial_values_expression = ("0.", "0.", "0.", "0.5*near(x[0],  0.) -0.5*near(x[0],  1.)"),
        boundary_conditions = [{"subspace": 0,
                "value_expression": ("0.", "0."), "degree": 3,
                "location_expression": "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)", "method": "topological"},
            {"subspace": 2,
                "value_expression": "0.5", "degree": 2, 
                "location_expression": "near(x[0],  0.)", "method": "topological"},
             {"subspace": 2,
                "value_expression": "-0.5", "degree": 2, 
                "location_expression": "near(x[0],  1.)", "method": "topological"}],
        start_time = 0.,
        end_time = 10.,
        time_step_size = 1.e-3,
        stop_when_steady = True,
        steady_relative_tolerance=1.e-4,
        adaptive = False,
        adaptive_metric = "all",
        adaptive_solver_tolerance = 1.e-4,
        nlp_absolute_tolerance = 1.e-8,
        nlp_relative_tolerance = 1.e-8,
        nlp_max_iterations = 50,
        automatic_jacobian = False,
        restart = False,
        restart_filepath = ""):
    """Run Phaseflow.
    
    Phaseflow is configured entirely through the arguments in this run() function.
    
    See the tests and examples for demonstrations of how to use this.
    """
    

    # Handle default function definitions.
    if m_B is None:
        
        def m_B(T, Ra, Pr, Re):
        
            return T*Ra/(Pr*Re**2)
    
    
    if dm_B is None:
        
        def dm_B(T, Ra, Pr, Re):

            return Ra/(Pr*Re**2)
    
    
    # Report arguments.
    phaseflow.helpers.print_once("Running Phaseflow with the following arguments:")
    
    phaseflow.helpers.print_once(phaseflow.helpers.arguments())
    
    phaseflow.helpers.mkdir_p(output_dir)
    
    if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
        
        arguments_file = open(output_dir + "/arguments.txt", "w")
        
        arguments_file.write(str(phaseflow.helpers.arguments()))

        arguments_file.close()
    
    
    # Check if 1D/2D/3D.
    dimensionality = mesh.type().dim()
    
    phaseflow.helpers.print_once("Running "+str(dimensionality)+"D problem")
    
    
    # Initialize time.
    if restart:
    
        with h5py.File(restart_filepath, "r") as h5:
            
            time = h5["t"].value
            
            assert(abs(time - start_time) < TIME_EPS)
    
    else:
    
        time = start_time
    
    
    # Define the mixed finite element and the solution function space.
    W_ele = make_mixed_fe(mesh.ufl_cell())
    
    W = fenics.FunctionSpace(mesh, W_ele)
    
    
    # Set the initial values.
    if restart:
            
        mesh = fenics.Mesh()
        
        with fenics.HDF5File(mesh.mpi_comm(), restart_filepath, "r") as h5:
        
            h5.read(mesh, "mesh", True)
        
        W_ele = make_mixed_fe(mesh.ufl_cell())
    
        W = fenics.FunctionSpace(mesh, W_ele)
    
        w_n = fenics.Function(W)
    
        with fenics.HDF5File(mesh.mpi_comm(), restart_filepath, "r") as h5:
        
            h5.read(w_n, "w")
    
    else:

        w_n = fenics.interpolate(fenics.Expression(initial_values_expression,
            element=W_ele), W)
            
        
    # Organize the boundary conditions.
    bcs = []
    
    for item in boundary_conditions:
    
        bcs.append(fenics.DirichletBC(W.sub(item["subspace"]), item["value_expression"],
            item["location_expression"], method=item["method"]))
    
    
    # Set the variational form.
    """Set local names for math operators to improve readability."""
    inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
    
    """The linear, bilinear, and trilinear forms b, a, and c, follow the common notation 
    for applying the finite element method to the incompressible Navier-Stokes equations,
    e.g. from danaila2014newton and huerta2003fefluids.
    """
    def b(u, q):
        return -div(u)*q  # Divergence
    
    
    def D(u):
    
        return sym(grad(u))  # Symmetric part of velocity gradient
    
    
    def a(mu, u, v):
        
        return 2.*mu*inner(D(u), D(v))  # Stokes stress-strain
    
    
    def c(w, z, v):
        
        return dot(dot(grad(z), w), v)  # Convection of the velocity field
    
    
    Delta_t = fenics.Constant(time_step_size)
    
    Re = fenics.Constant(reynolds_number)
    
    Ra = fenics.Constant(rayleigh_number)
    
    Pr = fenics.Constant(prandtl_number)
    
    Ste = fenics.Constant(stefan_number)
    
    g = fenics.Constant(gravity)
    
    def f_B(T): # Buoyancy force, $f = ma$
    
        return m_B(T=T, Ra=Ra, Pr=Pr, Re=Re)*g  
    
    
    gamma = fenics.Constant(penalty_parameter)
    
    T_r = fenics.Constant(regularization_central_temperature)
    
    r = fenics.Constant(regularization_smoothing_factor)
    
    def phi(T): # Regularized semi-phase-field
    
        return 0.5*(1. + fenics.tanh((T_r - T)/r))  
    
    
    def P(P_L, P_S, T):
    
        return P_L + (P_S - P_L)*phi(T)

    
    mu_L = fenics.Constant(liquid_viscosity)
    
    mu_S = fenics.Constant(solid_viscosity)
    
    def mu(T): # Variable viscosity
    
        return P(P_L = mu_L, P_S = mu_S, T = T)
    
    
    cp_L = fenics.Constant(liquid_heat_capacity)
    
    cp_S = fenics.Constant(solid_heat_capacity)

    def cp(T): # Variable heat capacity
    
        return P(P_L = cp_L, P_S = cp_S, T = T)
    
    
    k_L = fenics.Constant(liquid_thermal_conductivity)
    
    k_S = fenics.Constant(solid_thermal_conductivity)
    
    def k(T): # Variable thermal conductivity
    
        return P(P_L = k_L, P_S = k_S, T = T)
    
    
    u_n, p_n, T_n = fenics.split(w_n)

    psi_u, psi_p, psi_T = fenics.TestFunctions(W)
    
    w_m = fenics.Function(W)
    
    u_m, p_m, T_m = fenics.split(w_m)

    def d(f):
    
        return 1./(cp_L*Delta_t)*psi_T*(2*f(T_m)*cp(T_m) - f(T_m)*cp(T_n) - f(T_n)*cp(T_m))
    
    
    F = (
        b(u_m, psi_p) - gamma*psi_p*p_m
        + 1./Delta_t*dot(psi_u, u_m - u_n)
        + c(u_m, u_m, psi_u) + b(psi_u, p_m) + 1./mu_L*a(mu(T_m), u_m, psi_u)
        + dot(psi_u, f_B(T_m))
        + d(lambda T: T)
        - 1./cp_L*dot(u_m, grad(psi_T))*T_m*cp(T_m)
        + 1./(k_L*Pr)*dot(grad(psi_T), k(T_m)*grad(T_m))
        - 1./Ste*d(phi)
        )*fenics.dx

    dw = fenics.TrialFunction(W)
    
    if automatic_jacobian:
    
        JF = fenics.derivative(F, w_m, dw)
        
    else:
    
        du, dp, dT = fenics.split(dw)

        def df_B(T):
            
            return dm_B(T=T, Ra=Ra, Pr=Pr, Re=Re)*g
        
        
        def sech(theta):
        
            return 1./fenics.cosh(theta)
        
        
        def dphi(T):
        
            return -sech((T_r - T)/r)**2/(2.*r)
            
            
        def dP(P_S, P_L, T):
        
            return (P_S - P_L)*dphi(T_m)

            
        def dmu(T):
        
            return dP(P_L = mu_L, P_S = mu_S, T = T)
            
            
        def dcp(T):
        
            return dP(P_L = cp_L, P_S = cp_S, T = T)
            
        
        def dk(T):
        
            return dP(P_L = k_L, P_S = k_S, T = T)
            
            
        def Dd(f, df):
        
            return 1./(cp_L*Delta_t)*psi_T*dT* \
                (2.*(df(T_m)*cp(T_m) + f(T_m)*dcp(T_m)) - df(T_m)*cp(T_n) - f(T_n)*dcp(T_m))
        
        # Set the Jacobian (formally the Gateaux derivative) in variational form.
        JF = (
            b(du, psi_p) - gamma*dp*psi_p
            + 1./Delta_t*dot(psi_u, du) 
            + c(u_m, du, psi_u) + c(du, u_m, psi_u) 
            + b(psi_u, dp)
            + 1./mu_L*(a(dT*dmu(T_m), u_m, psi_u) + a(mu(T_m), du, psi_u))
            + dot(psi_u, dT*df_B(T_m))
            + Dd(f=lambda T: T, df=lambda T: 1.) 
            - 1./Ste*Dd(phi, dphi)
            + 1./(k_L*Pr)*dot(grad(psi_T), dT*dk(T_m)*grad(T_m) + k(T_m)*grad(dT))
            - 1./cp_L*(dot(du,grad(psi_T))*T_m*cp(T_m) 
                - dot(u_m,grad(psi_T))*(dT*T_m*dcp(T_m) + dT*cp(T_m)))
            )*fenics.dx

        
    # Set the functional metric for the error estimator for adaptive mesh refinement.
    """I haven't found a good way to make this flexible yet.
    Ideally the user would be able to write the metric, but this would require giving the user
    access to much data that phaseflow is currently hiding.
    """
    M = phi(T_m)*fenics.dx
    
    if adaptive_metric == "phase_only":
    
        pass
        
    elif adaptive_metric == "all":
        
        M += T_m*fenics.dx
        
        for i in range(dimensionality):
        
            M += u_m[i]*fenics.dx
            
    else:
        
        assert(False)
        
        
    # Make the problem.
    problem = fenics.NonlinearVariationalProblem(F, w_m, bcs, JF)
    
    
    # Make the solvers.
    """ For the purposes of this project, it would be better to just always use the adaptive solver; but
    unfortunately the adaptive solver encounters nan's whenever evaluating the error for problems not 
    involving phase-change. So far my attempts at writing a MWE to reproduce the  issue have failed.
    """   
    adaptive_solver = fenics.AdaptiveNonlinearVariationalSolver(problem, M)
    
    adaptive_solver.parameters["nonlinear_variational_solver"]["newton_solver"]["maximum_iterations"]\
        = nlp_max_iterations
    
    adaptive_solver.parameters["nonlinear_variational_solver"]["newton_solver"]["absolute_tolerance"]\
        = nlp_absolute_tolerance
    
    adaptive_solver.parameters["nonlinear_variational_solver"]["newton_solver"]["relative_tolerance"]\
        = nlp_relative_tolerance

    static_solver = fenics.NonlinearVariationalSolver(problem)
    
    static_solver.parameters["newton_solver"]["maximum_iterations"] = nlp_max_iterations
    
    static_solver.parameters["newton_solver"]["absolute_tolerance"] = nlp_absolute_tolerance
    
    static_solver.parameters["newton_solver"]["relative_tolerance"] = nlp_relative_tolerance
    
    
    # Open a context manager for the output file.
    with fenics.XDMFFile(output_dir + "/solution.xdmf") as solution_file:

    
        # Write the initial values.
        write_solution(solution_file, w_n, time) 

        if start_time >= end_time - TIME_EPS:
    
            phaseflow.helpers.print_once("Start time is already too close to end time. Only writing initial values.")
            
            return w_n, mesh
    
    
        # Solve each time step.
        progress = fenics.Progress("Time-stepping")
        
        fenics.set_log_level(fenics.PROGRESS)
        
        for it in range(1, MAX_TIME_STEPS):
            
            if(time > end_time - TIME_EPS):
                
                break
            
            if adaptive:
            
                adaptive_solver.solve(adaptive_solver_tolerance)
                
            else:
            
                static_solver.solve()
            
            time = start_time + it*time_step_size
            
            phaseflow.helpers.print_once("Reached time t = " + str(time))
            
            write_solution(solution_file, w_m, time)
            
            
            # Write checkpoint/restart files.
            restart_filepath = output_dir + "/restart_t" + str(time) + ".h5"
            
            with fenics.HDF5File(fenics.mpi_comm_world(), restart_filepath, "w") as h5:
                
                h5.write(mesh.leaf_node(), "mesh")
            
                h5.write(w_m.leaf_node(), "w")
                
            if fenics.MPI.rank(fenics.mpi_comm_world()) is 0:
            
                with h5py.File(restart_filepath, "r+") as h5:
                    
                    h5.create_dataset("t", data=time)
            
            
            # Check for steady state.
            if stop_when_steady and steady(W, w_m, w_n, steady_relative_tolerance):
            
                phaseflow.helpers.print_once("Reached steady state at time t = " + str(time))
                
                break
                
                
            # Set initial values for next time step.
            w_n.leaf_node().vector()[:] = w_m.leaf_node().vector()
            
            
            # Report progress.
            progress.update(time / end_time)
            
            if time >= (end_time - fenics.dolfin.DOLFIN_EPS):
            
                phaseflow.helpers.print_once("Reached end time, t = " + str(end_time))
            
                break
    
    
    # Return the interpolant to sample inside of Python.
    w_m.rename("w", "state")
    
    return w_m, mesh
    
    
if __name__=="__main__":

    run()
