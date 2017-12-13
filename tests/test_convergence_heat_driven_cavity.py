"""This module verifies the orders of convergence for the heat driven cavity benchmark."""
from .context import phaseflow

import fenics
import pytest


def verify_against_wang2010(w, mesh):

    data = {'Ra': 1.e6, 'Pr': 0.71, 'x': 0.5, 'y': [0., 0.15, 0.35, 0.5, 0.65, 0.85, 1.], 'ux': [0.0000, -0.0649, -0.0194, 0.0000, 0.0194, 0.0649, 0.0000]}
    
    bbt = mesh.bounding_box_tree()
    
    for i, true_ux in enumerate(data['ux']):
    
        p = fenics.Point(data['x'], data['y'][i])
        
        if bbt.collides_entity(p):
        
            wval = w(p)
        
            ux = wval[0]*data['Pr']/data['Ra']**0.5
        
            assert(abs(ux - true_ux) < 2.e-2)
            
    
output_dir = "output/test_wang2010_natural_convection_air/"

def heat_driven_cavity(nx, nt):
    
    def m_B(T, Ra, Pr, Re):
        
        return T*Ra/(Pr*Re**2)
    

    def ddT_m_B(T, Ra, Pr, Re):

        return Ra/(Pr*Re**2)

        
    phaseflow.run(output_dir = "output/convergence_heat_driven_cavity/nt" + str(nt)
            + "/nx" + str(nx) + "/",
        mesh = fenics.UnitSquareMesh(fenics.dolfin.mpi_comm_world(), nx, nx),
        rayleigh_number = 1.e6,
        prandtl_number = 0.71,
        stefan_number = 0.045,
        thermal_conductivity = 1.,
        gravity = (0., -1.),
        m_B = m_B,
        ddT_m_B = ddT_m_B,
        penalty_parameter = 1.e-7,
        temperature_of_fusion = -1.e12,
        initial_values_expression = ("0.", "0.", "0.",
            "0.5*near(x[0],  0.) -0.5*near(x[0],  1.)"),
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
        end_time = 0.02,
        time_step_size = dt,
        stop_when_steady = False,
        nlp_absolute_tolerance = 1.e-8,
        nlp_relative_tolerance = 1.e-8,
        nlp_max_iterations = 50,
        restart = False,
        restart_filepath = "")



@pytest.mark.dependency()
def test_wang2010_natural_convection_air():
    
    w, mesh = phaseflow.run(output_dir=output_dir, stop_when_steady=True)
        
    verify_against_wang2010(w, mesh)
    

@pytest.mark.dependency(depends=["test_wang2010_natural_convection_air"])
def test_wang2010_natural_convection_air_restart():

    w, mesh = phaseflow.run(restart = True,
        restart_filepath = output_dir+'restart_t0.067.h5',
        start_time = 0.067,
        output_dir=output_dir)
        
    verify_against_wang2010(w, mesh)
    
    
if __name__=='__main__':
    
    test_wang2010_natural_convection_air()
    
    test_wang2010_natural_convection_air_restart()
