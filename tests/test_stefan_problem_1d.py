from builtins import str
from builtins import range
from .context import phaseflow

import fenics
import scipy.optimize as opt

""" Melting data From Kai's MATLAB script"""
melting_data = [
    {'Ste': 1., 'time': 0.01, 'true_pci_pos': 0.075551957640682}, 
    {'Ste': 0.1, 'time': 0.01, 'true_pci_pos': 0.037826726426565},
    {'Ste': 0.01, 'time': 0.1, 'true_pci_pos': 0.042772111844781}] 
    
"""Solidification datat from MATLAB script solving Worster2000"""
solidification_data = [{'Ste': 0.125, 'time': 1., 'true_pci_pos': 0.49}] 


def get_pci_position(w):

    def theta(x):
        
        wval = w.leaf_node()(fenics.Point(x))
        
        return wval[2]
    
    pci_pos = opt.bisect(theta, 0.0001, 0.5)
    
    assert(not (pci_pos is None))
    

def verify_pci_position(true_pci_position, tolerance, w):
    
    pci_pos = get_pci_position(w)
    
    assert(abs(pci_pos - true_pci_position) < tolerance)
    

def refine_near_left_boundary(mesh, cycles):
    """ Refine mesh near the left boundary.
    The usual approach of using SubDomain and EdgeFunction isn't appearing to work
    in 1D, so I'm going to just loop through the cells of the mesh and set markers manually.
    """
    for i in range(cycles):
        
        cell_markers = fenics.CellFunction("bool", mesh)
        
        cell_markers.set_all(False)
        
        for cell in fenics.cells(mesh):
            
            found_left_boundary = False
            
            for vertex in fenics.vertices(cell):
                
                if fenics.near(vertex.x(0), 0.):
                    
                    found_left_boundary = True
                    
            if found_left_boundary:
                
                cell_markers[cell] = True
                
                break # There should only be one such point.
                
        mesh = fenics.refine(mesh, cell_markers)
        
    return mesh
    
    
def stefan_problem(output_dir = "output/test_stefan_problem_melt_Ste1/",
        Ste = 1.,
        theta_h = 1.,
        theta_c = -1.,
        r = 0.005,
        dt = 0.001,
        start_time = 0.,
        end_time = 0.01,
        initial_uniform_cell_count = 1,
        hot_boundary_refinement_cycles = 10,
        adaptive = False,
        restart = False,
        restart_filepath = ""):
    
    mesh = fenics.UnitIntervalMesh(initial_uniform_cell_count)
    
    mesh = refine_near_left_boundary(mesh, hot_boundary_refinement_cycles)
    
    w, mesh = phaseflow.run(
        output_dir = output_dir,
        prandtl_number = 1.,
        stefan_number = Ste,
        gravity = [0.],
        mesh = mesh,
        initial_values_expression = (
            "0.",
            "0.",
            "("+str(theta_h)+" - "+str(theta_c)+")*(x[0] <= 0.001) + "+str(theta_c)),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': [0.], 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_h, 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_c, 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': "topological"}],
        temperature_of_fusion = 0.,
        regularization_smoothing_factor = r,
        nlp_relative_tolerance = 1.e-8,
        adaptive = adaptive,
        adaptive_solver_tolerance = 1.e-8,
        time_step_size = dt,
        start_time = start_time,
        end_time = end_time)
        
    return w
        

def test_stefan_problem_melt_Ste0p045():
    
    w = stefan_problem(output_dir = "test_stefan_problem_melt_Ste0p045/",
        Ste=0.045, r=0.001, dt=1.e-3, start_time = 0., end_time = 0.1,
        theta_h = 1., theta_c = -0.01,
        initial_uniform_cell_count = 10000, hot_boundary_refinement_cycles = 0,
        restart = False, restart_filepath = "")
    
    """ Verify against solution from Kai's MATLAB script. """
    verify_pci_position(true_pci_position=0.0941, tolerance=2.e-2,  w=w)
    
    
def test_stefan_problem_melt_Ste0p045_convergence():

    for r in [0.01, 0.005, 0.0025]:
    
        for nx in [1000, 2000, 4000]:
    
            for nt in [1000, 2000, 4000]:
    
                w = stefan_problem(
                    output_dir = "test_stefan_problem_melt_Ste0p045/nt" + str(nt) + "/nx"
                        + str(nx) + "/r" + str(r) + "/",
                    Ste=0.045, r=r, dt=0.1/float(nt), start_time = 0., end_time = 0.1,
                    theta_h = 1., theta_c = -0.01,
                    initial_uniform_cell_count = nx, hot_boundary_refinement_cycles = 0,
                    restart = False, restart_filepath = "")


def test_stefan_problem_melt_Ste0p045_report_pci():
    
    output_dir = "test_stefan_problem_melt_Ste0p045/"
    
    start_time = 0.
    
    with open("pci_melt_Ste0p045.txt", "w") as pci_file:

        pci_file.write("t, x_pci\n")
        
        for end_time in [0.001, 0.01, 0.1]:
        
            if start_time == 0.:
            
                restart = False
                
                restart_filepath = ""
                
            else:
            
                restart = True
                
                restart_filepath = output_dir + "restart_t" + str(start_time) + ".h5"
            
            w = stefan_problem(output_dir = output_dir,
                Ste=0.045, r=0.01, dt=1.e-3, start_time = start_time, end_time = end_time,
                theta_h = 1., theta_c = -0.01,
                initial_uniform_cell_count = 10000, hot_boundary_refinement_cycles = 0,
                restart = restart, restart_filepath = restart_filepath)
        
            pci_file.write(str(end_time) + ", " + str(get_pci_position(w)) + "\n")
            
            start_time = end_time
    
    """ Verify against solution from Kai's MATLAB script. """
    verify_pci_position(true_pci_position=0.0941, tolerance=2.e-2,  w=w)
    

def test_stefan_problem_solidify_Ste0p125(Ste = 0.125,
        theta_h = 0.01,
        theta_c = -1.,
        theta_f = 0.,
        r = 0.01,
        dt = 0.01,
        end_time = 1.,
        nlp_absolute_tolerance = 1.e-4,
        initial_uniform_cell_count = 100,
        cool_boundary_refinement_cycles = 0):
    
    mesh = fenics.UnitIntervalMesh(initial_uniform_cell_count)
    
    mesh = refine_near_left_boundary(mesh, cool_boundary_refinement_cycles)
    
    w, mesh = phaseflow.run(
        output_dir = "output/test_stefan_problem_solidify/dt" + str(dt) + 
            "/tol" + str(nlp_absolute_tolerance) + "/",
        prandtl_number = 1.,
        stefan_number = Ste,
        gravity = [0.],
        mesh = mesh,
        initial_values_expression = (
            "0.",
            "0.",
            "("+str(theta_c)+" - "+str(theta_h)+")*near(x[0],  0.) + "+str(theta_h)),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': [0.], 'degree': 3, 'location_expression': "near(x[0],  0.) | near(x[0],  1.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_c, 'degree': 2, 'location_expression': "near(x[0],  0.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_h, 'degree': 2, 'location_expression': "near(x[0],  1.)", 'method': "topological"}],
        temperature_of_fusion = theta_f,
        regularization_smoothing_factor = r,
        nlp_absolute_tolerance = nlp_absolute_tolerance,
        adaptive = True,
        adaptive_solver_tolerance = 1.e-8,
        end_time = end_time,
        time_step_size = dt)
    
    """ Verify against solution from MATLAB script solving Worster2000. """
    verify_pci_position(true_pci_position=0.49, r=r, w=w)

    
if __name__=='__main__':
    
    test_stefan_problem_melt_Ste0p045()
    
    test_stefan_problem_solidify_Ste0p125()
    