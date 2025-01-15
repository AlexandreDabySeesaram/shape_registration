

import dolfin
import numpy as np
import matplotlib.pyplot as plt
import dolfin_mech as dmech
from shape_derivatives import *



#%% Load image
print("* Loading the image")
img_raw = plt.imread("lung_blurred.pgm")
img = np.array(img_raw)
img[img>150] = 0                                                                                # Remove left lung
img = -1*img+50                                                                                 # Set inside lung negative and outside positive

(Nx, Ny) = img.shape
mesh = dolfin.UnitSquareMesh(Nx, Ny, "crossed")                                                        # Create a mesh at the image dimensions


class FE_image_self(dolfin.UserExpression):
    def eval_cell(self, value, x, ufc_cell):
        p = dolfin.Cell(mesh, ufc_cell.index).midpoint()
        i, j = int(p[0]*(Nx-1)), int(p[1]*(Ny-1))
        value[:] = img[-(j+1), i]

    def value_shape(self):
        return ()

img_expr = FE_image_self()                                                                      # Get an dolfin expression for the image

V = dolfin.FunctionSpace(mesh, "Lagrange", 1)

image = dolfin.interpolate(img_expr, V)                                                                # Project it onto the image mesh


image.set_allow_extrapolation(True)                                                             


dmech.write_VTU_file(
    filebasename = "image_lung",
    function = image,
    time = 0,
    preserve_connectivity = True)


#%% Create the domain \omega

circle_center = (0.5, 0.5)                                      # Center of the circle
radius = 0.35                                                   # Radius of the circle

import mshr 
center = dolfin.Point(circle_center[0], circle_center[1])              # Center of the disc
resolution = 50                                                 # Resolution of the mesh
domain = mshr.Circle(center, radius)
mesh_omega = mshr.generate_mesh(domain, resolution)

integral_value = int_I(mesh_omega, image)   

print(f"Integral of f over the domain: {integral_value}")

# plot(I(mesh_omega, image))                                      # Plot image projected onto the initial domain


# %% Naive gradient descent

# Algorithms parameters
maxit = 500                                                     # max number of iteration
step = 0.1                                                      # initial step size
coeffStep = 1.5                                                 # step increase factor at each iteration ( > 1)
minStep = 1e-9                                                  # minimum step size (stop criterion)

# Shape derivative parameters
alpha = 10                                                       # dissipation term
gamma = 1                                                       # preserve mesh quality (arrondis aussi les angles...)

# Initialization


u_fs            = dolfin.VectorFunctionSpace(mesh_omega, "CG", 1)
u               = dolfin.Function(u_fs, name="mapping")
u.vector()[:]   = 0
dolfin.ALE.move(mesh_omega, u)


loss_vect = [int_I(mesh_omega, image)]

# Optimization loop (gradient descent)
k = 0


filebasename = "mapping_lung_2D_H1_weight_L2_2"


dmech.write_VTU_file(
filebasename = filebasename,
function = u,
time = k,
preserve_connectivity = True)

while k<maxit and step >= minStep:
    k += 1
    # shape derivative computation and update gradient
    shape_gradient = shape_derivative_volume(mesh_omega, u, I(mesh_omega, image), grad_I(mesh_omega, image), alpha = alpha, gamma = gamma)
    u, loss , step = update_GD(mesh_omega, image, u, -shape_gradient, step = step * coeffStep, minStep = minStep)
    # Print and store result
    print(f"it = {k}  |  loss = {loss:.10e}    ", end = "\r")
    loss_vect.append(loss)
    dmech.write_VTU_file(
    filebasename = filebasename,
    function = u,
    time = k,
    preserve_connectivity = True)
# %%
