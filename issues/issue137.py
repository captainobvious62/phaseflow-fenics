""" This is a MWE to demonstrate a bug
where adaptivenonlinearvariationalsolver produces NaN's
"""
import fenics


pressure_degree = 1

temperature_degree = 1

Re = 1.

mu = 1.

C = 1.

K = 1.

gamma = 1.e-7


def make_mixed_fe(mesh):
    """ Define mixed FE function space for the variational form."""
    
    velocity_degree = pressure_degree + 1

    ''' MixedFunctionSpace used to be available but is now deprecated. 
    The way that fenics separates function spaces and elements is confusing.
    To create the mixed space, I'm using the approach from https://fenicsproject.org/qa/11983/mixedfunctionspace-in-2016-2-0
    '''
    velocity_element = fenics.VectorElement('P', mesh.ufl_cell(), velocity_degree)

    pressure_element = fenics.FiniteElement('P', mesh.ufl_cell(), pressure_degree)

    temperature_element = fenics.FiniteElement('P', mesh.ufl_cell(), temperature_degree)

    solution_element = fenics.MixedElement([velocity_element, pressure_element, temperature_element])

    solution_function_space = fenics.FunctionSpace(mesh, solution_element)  
    
    return solution_function_space, solution_element

    
def write_solution(solution_file, w, time):
    """Write the solution to disk."""
    print("Writing solution to HDF5+XDMF")
    
    velocity, pressure, temperature = w.leaf_node().split()
    
    velocity.rename("u", "velocity")
    
    pressure.rename("p", "pressure")
    
    temperature.rename("T", "temperature")
    
    for var in [velocity, pressure, temperature]:
    
        solution_file.write(var, time)

        
def forms(W, w_k, w_n, dt, rayleigh_number, prandtl_number, gravity):
    """ Define the variational forms. """
    Ra = fenics.Constant(rayleigh_number)
    
    Pr = fenics.Constant(prandtl_number)
    
    g = fenics.Constant(gravity)
    
    inner, dot, grad, div, sym = fenics.inner, fenics.dot, fenics.grad, fenics.div, fenics.sym
    
    D = lambda u : sym(grad(u))
    
    a = lambda mu, u, v : 2.*mu*inner(D(u), D(v))
    
    b = lambda u, q : -div(u)*q
    
    c = lambda w, z, v : dot(dot(grad(z), w), v)
    
    f_B = lambda T : g*T*Ra/(Pr*Re**2)
    
    u_n, p_n, T_n = fenics.split(w_n)

    w_w = fenics.TrialFunction(W)
    
    u_w, p_w, T_w = fenics.split(w_w)
    
    v, q, phi = fenics.TestFunctions(W)
    
    u_k, p_k, T_k = fenics.split(w_k)

    F = (
        b(u_k, q) - gamma*p_k*q
        + dot(u_k - u_n, v)/dt
        + c(u_k, u_k, v) + b(v, p_k) + a(mu, u_k, v)
        + dot(f_B(T_k), v)
        + C/dt*(T_k - T_n)*phi
        - dot(C*T_k*u_k, grad(phi)) 
        + K/Pr*dot(grad(T_k), grad(phi))
        )*fenics.dx
    
    
    # Set the Gateaux derivative.
    ddT_f_B = lambda T : g*Ra/(Pr*Re**2)

    JF = (
        b(u_w, q) - gamma*p_w*q 
        + dot(u_w, v)/dt
        + c(u_k, u_w, v) + c(u_w, u_k, v) + b(v, p_w)
        + a(mu, u_w, v) 
        + dot(T_w*ddT_f_B(T_k), v)
        + C/dt*T_w*phi
        - dot(C*T_k*u_w, grad(phi))
        - dot(C*T_w*u_k, grad(phi))
        + K/Pr*dot(grad(T_w), grad(phi))
        )*fenics.dx

        
    # Set goal functional for adaptive mesh refinement.
    M = (u_k[0] + u_k[1] + T_k)*fenics.dx
    
    
    # Return the nonlinear variational form, its Gateaux derivative, and the AMR metric
    return F, JF, M


def adaptive_natural_convection_in_differentially_heated_cavity():
    
    m = 20
    
    mesh = fenics.UnitSquareMesh(m, m, 'crossed')
    
    W, W_ele = make_mixed_fe(mesh)
    
    w_k = fenics.Function(W)
    
    T_hot, T_cold = 0.5, -0.5
    

    # Set initial values.
    w_n = fenics.Expression(("0.", "0.", "0.",
        str(T_hot)+"*near(x[0],  0.) "+str(T_cold)+"*near(x[0],  1.)"), element=W_ele)
    
    
    # Set boundary conditions.
    bcs = [
        fenics.DirichletBC(W.sub(0), ("0.", "0."), 
            "near(x[0],  0.) | near(x[0],  1.) | near(x[1], 0.) | near(x[1],  1.)",
            method='topological'),
        fenics.DirichletBC(W.sub(2), str(T_hot), "near(x[0],  0.)", method='topological'),
        fenics.DirichletBC(W.sub(2), str(T_cold), "near(x[0],  1.)", method='topological')]
        
    
    # Make the problem.
    time_step_size = 1.e-3
    
    dt = fenics.Constant(time_step_size)
    
    F, JF, M = forms(W=W, w_k=w_k, w_n=w_n, dt=dt, rayleigh_number=1.e6, prandtl_number=0.71,
        gravity = (0., -1.))
    
    problem = fenics.NonlinearVariationalProblem(F, w_k, bcs, JF)
    
    
    # Solve a sequence of time steps.
    time, end_time = 0., 10.
    
    solver = fenics.AdaptiveNonlinearVariationalSolver(problem, M)
    
    w_n = fenics.interpolate(w_n, W)

    progress = fenics.Progress('Time-stepping')

    fenics.set_log_level(fenics.PROGRESS)
        
    with fenics.XDMFFile('output/solution.xdmf') as solution_file:
    
        # Write the initial values
        write_solution(solution_file, w_n, time)
            
        while time < (end_time - fenics.DOLFIN_EPS):
        
            solver.solve(1.e-4)
            
            time += time_step_size
            
            write_solution(solution_file, w_k, time)
            
            progress.update(time / end_time)
            
            
            # Double the time step size, which had to be small in the beginning.
            time_step_size *= 2
            
            dt.assign(time_step_size)
            
            
            # Update initial values for the next time step.
            w_n.leaf_node().vector()[:] = w_k.leaf_node().vector()  
    
    
if __name__=='__main__':
    
    adaptive_natural_convection_in_differentially_heated_cavity()
    