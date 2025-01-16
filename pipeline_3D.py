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

#%% Load image

image_basename  = "PA5_Binary"
image_suffix    = "signed_int_3"
result_folder   = "Results/" 
filebasename    = result_folder+"mapping_lung_3D_4"
# image_name      = "PA5_Binary_signed_int.vti" 
image_name      = image_basename+"_"+image_suffix+".vti"
image_folder    = "Images/"
image_path      = image_folder+image_name

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


#%% Tracking


# Solver parameters
maxit           = 500                                                                       # max number of iteration
step            = 0.01                                                                      # initial step size
coeffStep       = 1.5                                                                       # step increase factor at each iteration ( > 1)
minStep         = 1e-9                                                                      # minimum step size (stop criterion)

# Shape derivative parameters
alpha           = 1e-3                                                                      # weight L1 term of H1 norm


# Initialization

mesh_Omega_0    = dolfin.Mesh(mesh_omega)                                                   # Reference configuration mesh
u_fs_Omega_0    = dolfin.VectorFunctionSpace(mesh_Omega_0, "CG", 1)                         # d-D vector space defined on reference configuration  
u_Omega_0       = dolfin.Function(u_fs_Omega_0, name="mapping")                             # Mapping defined on the reference configuration mesh


u_fs            = dolfin.VectorFunctionSpace(mesh_omega, "CG", 1)                           # d-D vector space defined on current configuration                             
u               = dolfin.Function(u_fs, name="mapping")                                     # Mapping defined on the current configuration mesh

loss_vect       = [int_I(mesh_omega, Img_3D_expr)]                                          # Store the evolution of the loss function

# Optimization loop ( naive gradient descent)

k = 0
dmech.write_VTU_file(
    filebasename            = filebasename,
    function                = u,
    time                    = k,
    preserve_connectivity   = True)

while k<maxit and step >= minStep:
    k += 1
    # shape derivative computation and update
    shape_gradient = shape_derivative_volume(
                        mesh        = mesh_omega, 
                        I           = proj_I(mesh_omega, Img_3D_expr), 
                        grad_I      = grad_I(mesh_omega, Img_3D_expr), 
                        alpha       = alpha)

    u, loss , step = update_GD(
                        mesh        = mesh_omega, 
                        image       = Img_3D_expr, 
                        u           = u, 
                        descentDir  = -shape_gradient, 
                        step        = step * coeffStep, 
                        minStep     = minStep)

    # Print and store result
    print(f"* iteration = {k}  |  loss = {loss:.10e}    ", end = "\n")
    loss_vect.append(loss)

    dmech.write_VTU_file(
        filebasename            = filebasename,
        function                = u,
        time                    = k,
        preserve_connectivity   = True)





