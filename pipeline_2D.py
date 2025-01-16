

import dolfin
import numpy as np
import matplotlib.pyplot as plt
import dolfin_mech as dmech
from shape_derivatives import *

#%% Domain omega generation

circle_center = (0.5, 0.5)                                                                  # Center of the circle
radius = 0.35                                                                               # Radius of the circle

import mshr 
center      = dolfin.Point(circle_center[0], circle_center[1])                              # Center of the disc
resolution  = 50                                                                            # Resolution of the mesh
domain      = mshr.Circle(center, radius)
mesh_omega  = mshr.generate_mesh(domain, resolution)



#%% Load image


result_folder   = "Results/" 
filebasename    = result_folder+"mapping_lung_2D_interpol"
image_name      = "lung_blurred.pgm" 
image_folder    = "Images/"
image_path      = image_folder+image_name



print("* Loading the image")
img_raw         = plt.imread(image_path)
img             = np.array(img_raw)
img[img>150]    = 0                                                                         # Remove left lung
img             = -1*img+50                                                                 # Set inside lung negative and outside positive (should automate depending on max value in image)

(Nx, Ny) = img.shape
mesh = dolfin.UnitSquareMesh(Nx, Ny, "crossed")                                             # Create a mesh at the image dimensions


class FE_image_self(dolfin.UserExpression):
    def eval_cell(self, value, x, ufc_cell):
        p = dolfin.Cell(mesh, ufc_cell.index).midpoint()
        i, j = int(p[0]*(Nx-1)), int(p[1]*(Ny-1))
        value[:] = img[-(j+1), i]

    def value_shape(self):
        return ()

img_expr = FE_image_self()                                                                      # Get an dolfin expression for the image

V = dolfin.FunctionSpace(mesh, "Lagrange", 1)
image = dolfin.interpolate(img_expr, V)                                                         # Project it onto the image mesh
image.set_allow_extrapolation(True)                                                             


dmech.write_VTU_file(
    filebasename = image_folder+"image_lung_2D",
    function = image,
    time = 0,
    preserve_connectivity = True)





#%% Tracking


# Solver parameters
maxit           = 500                                                                           # max number of iteration
step            = 0.1                                                                           # initial step size
coeffStep       = 1.5                                                                           # step increase factor at each iteration ( > 1)
minStep         = 1e-9                                                                          # minimum step size (stop criterion)

# Shape derivative parameters
alpha           = 10                                                                            # weight L1 term of H1 norm


# Initialization

mesh_Omega_0    = dolfin.Mesh(mesh_omega)                                                       # Reference configuration mesh
u_fs_Omega_0    = dolfin.VectorFunctionSpace(mesh_Omega_0, "CG", 1)                             # d-D vector space defined on reference configuration  
u_Omega_0       = dolfin.Function(u_fs_Omega_0, name="mapping")                                 # Mapping defined on the reference configuration mesh


u_fs            = dolfin.VectorFunctionSpace(mesh_omega, "CG", 1)                               # d-D vector space defined on current configuration                             
u               = dolfin.Function(u_fs, name="mapping")                                         # Mapping defined on the current configuration mesh

loss_vect       = [int_I(mesh_omega, image)]                                                    # Store the evolution of the loss function

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
                        I           = proj_I(mesh_omega, image), 
                        grad_I      = grad_I(mesh_omega, image), 
                        alpha       = alpha)

    u, loss , step = update_GD(
                        mesh        = mesh_omega, 
                        image       = image, 
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




