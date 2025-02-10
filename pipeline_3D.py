import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import dolfin
import dolfin_warp as dwarp
import mshr
import dolfin_mech as dmech
from shape_derivatives import *
import create_data
import glob


#%% Domain omega generation

sphere_center   = (120, 120, 120)                                                           # Center of the sphere
sphere_radius   = 75                                                                        # Radius of the sphere
resolution      = 30                                                                        # Resolution of the mesh

# Create a 3D spherical domain
center          = dolfin.Point(sphere_center[0], sphere_center[1], sphere_center[2])        # Center of the disc
radius          = 1.4*sphere_radius                                                         # Radius of the disc
domain          = mshr.Sphere(center, radius)
mesh_omega      = mshr.generate_mesh(domain, resolution)
mesh_omega.num_vertices()


#%% load previous lung mesh


# mesh_omega=dolfin.Mesh("Meshes/mesh_RL.xml")


#%% Load image

mesh_folder     = "Meshes"
mesh_name       = "3D_lung_PA5"
image_basename  = "PA5_Binary"
image_suffix    = "signed_int"
result_folder   = "Results/" 
filebasename    = result_folder+"Euler_back2normal_new_int"
mappingname     = result_folder+"Compare_pulled-back_mapping"
image_name      = image_basename+"_"+image_suffix+".vti"
image_folder    = "Images/"
image_path      = image_folder+image_name


# Checks folders existence
import os 
if not os.path.isdir(image_folder):
    os.system("mkdir "+image_folder)
    print("folder "+image_folder+" created")

if not os.path.isdir(result_folder):
    os.system("mkdir "+result_folder)
    print("folder "+result_folder+" created")

# Check if signed image exist
image_file = glob.glob(image_path)

if not image_file:
    create_data.sign_masking_binary(
        input_name          = image_folder+image_basename   , 
        suffix              = image_suffix                  , 
        scalar2zero         = 200                           ,
        scalar_background   = 0                             ,                               # Initial background pixel intensity
        scalar_foreground   = 100                           ,                               # Initial foreground pixel intensity
        target_value_bg     = 1                             ,                               # target background pixel intensity
        target_value_fg     = -1                            ,                               # target foreground pixel intensity
        target_type         = "signed_char"                 ,                               # unsigned_char, signed_char, float
        )


# Image expression in cpp

fe = dolfin.FiniteElement(
    family="Quadrature",
    cell=mesh_omega.ufl_cell(),
    quad_scheme="default")
name, cpp = dwarp.get_ExprIm_cpp_pybind(
    im_dim=3,
    im_type="im",
    verbose=0)
module = dolfin.compile_cpp_code(cpp)
expr = getattr(module, name)
Img_3D_expr = dolfin.CompiledExpression(
    expr(),
    element=fe)
Img_3D_expr.init_image(
    filename=image_path)

name, cpp = dwarp.get_ExprIm_cpp_pybind(
    im_dim=3,
    im_type="grad",
    verbose=0)
module = dolfin.compile_cpp_code(cpp)
expr = getattr(module, name)
Gradi_Img_3D_expr = dolfin.CompiledExpression(
    expr(),
    element=fe)
Gradi_Img_3D_expr.init_image(
    filename=image_path)


#%% Tracking

# Output settings
write_deformed_mesh     = False                                                              # Boolean export deformed mesh
print_iterations        = True                                                              # Boolean print gradient descent iterations
write_mapping           = True                                                              # Boolean mapping on initial mesh

# Solver parameters
maxit                   = 15                                                               # max number of iteration
step                    = 1e-2                                                         # initial step size
coeffStep               = 1.5                                                               # step increase factor at each iteration ( > 1)
minStep                 = 1e-9                                                              # minimum step size (stop criterion)

# Shape derivative parameters
alpha                   = 1e-3                                                              # weight L2 term of H1 norm


# Initialization

if write_mapping:
    mesh_Omega_0            = dolfin.Mesh(mesh_omega)                                       # Reference configuration mesh
    u_fs_Omega_0            = dolfin.VectorFunctionSpace(mesh_Omega_0, "CG", 1)             # d-D vector space defined on reference configuration  
    u_Omega_0               = dolfin.Function(u_fs_Omega_0, name="mapping")                 # Mapping defined on the reference configuration mesh

    #DEBUG
    I = dolfin.Identity(3)
    F = I + dolfin.grad(u_Omega_0)
    J = dolfin.det(F)
    Q_2 = dolfin.FunctionSpace(mesh_Omega_0, "DG", 0)

    #DEBUG END

u_fs                    = dolfin.VectorFunctionSpace(mesh_omega, "CG", 1)                   # d-D vector space defined on current configuration                             
u                       = dolfin.Function(u_fs, name="mapping")                             # Mapping defined on the current configuration mesh

loss_vect               = [int_I(mesh_omega, Img_3D_expr)]                                  # Store the evolution of the loss function

# Optimization loop ( naive gradient descent)

# Start by deleting all previous solution with same name
import os
os.system(f"rm {filebasename}*")


k = 0
dmech.write_VTU_file(
    filebasename            = filebasename  ,
    function                = u             ,
    time                    = k             ,
    preserve_connectivity   = True)

import time
t_start = time.time()

while k<maxit and step >= minStep:
    k += 1
    # shape derivative computation and update
    shape_gradient = shape_derivative_volume(
                        mesh        = mesh_omega                        , 
                        # I           = proj_I(mesh_omega, Img_3D_expr)   , 
                        grad_I      = grad_I(mesh_omega, Img_3D_expr)   , 
                        I           = Img_3D_expr                       , # No projection
                        # grad_I      = Gradi_Img_3D_expr                 , # No projection
                        alpha       = alpha)


    u, loss , step = update_GD(
                        mesh        = mesh_omega                        , 
                        image       = Img_3D_expr                       , 
                        u           = u                                 , 
                        descentDir  = -shape_gradient                   , 
                        step        = step * coeffStep                  , 
                        minStep     = minStep)

    # Print and store result
    print(f"* iteration = {k}  |  loss = {loss:.10e}    |  step = {step:.4e}    ", end = "\n")
    loss_vect.append(loss)
    if print_iterations:
        dmech.write_VTU_file(
            filebasename            = filebasename  ,
            function                = u             ,
            time                    = k             ,
            preserve_connectivity   = True)
        
    # print jacobian
    u_Omega_0.vector()[:]           = u.vector()[:]
    F_proj_2 = dolfin.project(J, Q_2)
    print(f"min jacobian is {min(F_proj_2.vector()[:])}")


if write_deformed_mesh:
    if not os.path.isdir(mesh_folder):
        os.system("mkdir "+mesh_folder)
        print("folder "+mesh_folder+" created")
    dolfin.File(mesh_folder+"/"+mesh_name+".xml") << mesh_omega

if write_mapping:
    u_Omega_0.vector()[:]           = u.vector()[:]
    dmech.write_VTU_file(
            filebasename            = mappingname   ,
            function                = u_Omega_0     ,
            time                    = 0             ,
            preserve_connectivity   = True)
    print("Done writing ref mapping")


t_stop = time.time()

print(f"* Duration (s) = {(t_stop-t_start):.4e}")
