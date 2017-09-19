from .context import phaseflow

import fenics


def binary_alloy_solidification(Ste = 0.125,
        T_h = -20.,
        T_c = -30.,
        C_0 = 0.1,
        T_f = -28.,
        D = 1./80.,
        initial_frozen_layer_width = 0.01,
        r = 0.001,
        dt = 1.,
        initial_uniform_cell_count = 1000,
        nlp_absolute_tolerance = 1.e-4,
        end_time = 3.,
        cool_boundary_refinement_cycles = 0,
        max_pci_refinement_cycles_per_time = 0,
        output_dir = 'output/binary_alloy/',
        automatic_jacobian = False):

    mesh = fenics.UnitIntervalMesh(initial_uniform_cell_count)
    
    ''' Refine mesh near hot boundary
    The usual approach of using SubDomain and EdgeFunction isn't appearing to work
    in 1D, so I'm going to just loop through the cells of the mesh and set markers manually.
    '''
    for i in range(cool_boundary_refinement_cycles):
    
        cell_markers = fenics.CellFunction("bool", mesh)
        
        cell_markers.set_all(False)
        
        for cell in fenics.cells(mesh):
        
            found_cool_boundary = False
        
            for vertex in fenics.vertices(cell):
            
                if fenics.near(vertex.x(0), 0., fenics.dolfin.DOLFIN_EPS):
                
                    found_cool_boundary = True
                    
            if found_cool_boundary:
            
                cell_markers[cell] = True

                break # There should only be one such point.
                
        mesh = fenics.refine(mesh, cell_markers)

    T_ref = T_f
    
    theta_h = (T_h - T_ref)/(T_h - T_c)
    
    theta_c = (T_c - T_ref)/(T_h - T_c)
    
    theta_f = (T_f - T_ref)/(T_h - T_c)

    w, mesh = phaseflow.run(
        output_dir = output_dir,
        Pr = 1.,
        Ste = Ste,
        Sc = 1.,
        K = 1.,
        D = D,
        g = [0.],
        mesh = mesh,
        max_pci_refinement_cycles_per_time = max_pci_refinement_cycles_per_time,
        initial_values_expression = (
            "0.",
            "0.",
            "("+str(theta_c)+" - "+str(theta_h)+")*(x[0] <= "+str(initial_frozen_layer_width)+") + "+str(theta_h),
            "(x[0] > "+str(initial_frozen_layer_width)+")*"+str(C_0)),
        boundary_conditions = [
            {'subspace': 0, 'value_expression': [0.], 'degree': 3,
                'location_expression': "near(x[0],  0.) | near(x[0],  1.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_c, 'degree': 2,
                'location_expression': "near(x[0],  0.)", 'method': "topological"},
            {'subspace': 2, 'value_expression': theta_h, 'degree': 2,
                'location_expression': "near(x[0],  1.)", 'method': "topological"},
            {'subspace': 3, 'value_expression': "0.", 'degree': 2,
                'location_expression': "near(x[0],  0.)", 'method': "topological"},
            {'subspace': 3, 'value_expression': str(C_0), 'degree': 2,
                'location_expression': "near(x[0],  1.)", 'method': "topological"}],
        regularization = {'theta_f': theta_f, 'r': r},
        nlp_absolute_tolerance = nlp_absolute_tolerance,
        nlp_max_iterations = 50,
        end_time = end_time,
        time_step_bounds = dt,
        automatic_jacobian = automatic_jacobian)
        
    return w, mesh


def test_binary_alloy_solidification():

    '''Vary the concentration diffusion'''
    
    for D in [1., 0.1, 0.01, 1.e-4, 1.e-8]:
    
        w, mesh = binary_alloy_solidification(Ste = 1.,
            T_h = 0.5,
            T_c = -0.5,
            T_f = 0.,
            C_0 = 0.1,
            D = D,
            r = 0.01,
            dt = 0.1,
            end_time = 0.3,
            initial_uniform_cell_count = 100,
            output_dir = 'output/binary_alloy/D'+str(D)+'/dt0.1/')
    
    '''Vary the time step size'''
    w, mesh = binary_alloy_solidification(Ste = 1.,
        T_h = 0.5,
        T_c = -0.5,
        T_f = 0.,
        C_0 = 0.1,
        D = 0.01,
        r = 0.01,
        dt = 0.05,
        end_time = 0.15,
        initial_uniform_cell_count = 100,
        output_dir = 'output/binary_alloy/D0.01/dt0.05')
    
    w, mesh = binary_alloy_solidification(Ste = 1.,
        T_h = 0.5,
        T_c = -0.5,
        T_f = 0.,
        C_0 = 0.1,
        D = 0.01,
        r = 0.01,
        dt = 0.01,
        end_time = 0.03,
        initial_uniform_cell_count = 100,
        output_dir = 'output/binary_alloy/D0.01/dt0.01')
    
if __name__=='__main__':
    
    test_binary_alloy_solidification()

