

from dolfin import *
import numpy as np
import matplotlib.pyplot as plt

import dolfin_mech as dmech
from shape_derivatives import *

#%% Synthetic image generation

img_raw = plt.imread("lung_blurred.pgm")
img = np.array(img_raw)
img[img>150] = 0

img = -1*img+50

(Nx, Ny) = img.shape
mesh = UnitSquareMesh(Nx, Ny, "crossed")

class FE_image_self(UserExpression):
    def __init__(self,image_name, image_ext,**kwargs):
        super().__init__(**kwargs)
        self.img = plt.imread(image_name+"."+image_ext)
        (self.Nx, self.Ny) = self.img.shape
        self.mesh = UnitSquareMesh(self.Nx, self.Ny, "crossed")

    def eval_cell(self, value, x, ufc_cell):
        p = Cell(self.mesh, ufc_cell.index).midpoint()
        i, j = int(p[0]*(self.Nx-1)), int(p[1]*(self.Ny-1))
        value[:] = self.img[-(j+1), i]

    def value_shape(self):
        return ()

class FE_image_self(UserExpression):

    def eval_cell(self, value, x, ufc_cell):
        p = Cell(mesh, ufc_cell.index).midpoint()
        i, j = int(p[0]*(Nx-1)), int(p[1]*(Ny-1))
        value[:] = img[-(j+1), i]

    def value_shape(self):
        return ()

y = FE_image_self()

# img = FE_image(
#     image_name="lung", 
#     image_ext="pgm"
# )

# Create the scalar function space for the image field
V = FunctionSpace(mesh, "Lagrange", 1)

# Create an instance of the FE_image and interpolate onto V
image = interpolate(y, V)


image.set_allow_extrapolation(True)


dmech.write_VTU_file(
    filebasename = "image_lung",
    function = image,
    time = 0,
    preserve_connectivity = True)







#%%

square_size = 1.0  # Length of the sides of the square
circle_center = (0.5, 0.5)  # Center of the circle
circle_radius = 0.25  # Radius of the circle

# mesh_omega = UnitSquareMesh(100, 100)

# Create a disc mesh
from mshr import *
center = Point(circle_center[0], circle_center[1])  # Center of the disc
radius = 1.4*circle_radius         # Radius of the disc
resolution = 50                   # Resolution of the mesh
domain = Circle(center, radius)
mesh_omega = generate_mesh(domain, resolution)

integral_value = int_I(mesh_omega, image)

print(f"Integral of f over the domain: {integral_value}")

plot(I(mesh_omega, image))

dmech.write_VTU_file(
    filebasename = "I_lung",
    function = I(mesh_omega, image),
    time = 0,
    preserve_connectivity = True)



# %% Naive gradient descent


# Algorithms parameters
maxit = 500           # max number of iteration
step = 0.5            # initial step size
coeffStep = 1.5       # step increase factor at each iteration ( > 1)
minStep = 1e-9        # minimum step size (stop criterion)

# Shape derivative parameters
alpha = 1            # dissipation term
gamma = 1            # preserve mesh quality (arrondis aussi les angles...)

# Initialization

mesh_Omega_0 = Mesh(mesh_omega)
u_fs_Omega_0 = VectorFunctionSpace(mesh_Omega_0, "CG", 1)
u_Omega_0 = Function(u_fs_Omega_0, name="mapping")


u_fs = VectorFunctionSpace(mesh_omega, "CG", 1)
u = Function(u_fs, name="mapping")
u.vector()[:] = 0
ALE.move(mesh_omega, u)


loss_vect = [int_I(mesh_omega, image)]

# Optimization loop (gradient descent)
k = 0

dmech.write_VTU_file(
filebasename = "mapping_lung_2D",
function = u_Omega_0,
time = k,
preserve_connectivity = True)

while k<maxit and step >= minStep:
    k += 1
    # shape derivative computation and update
    shape_gradient = shape_derivative_volume(mesh_omega, u, I(mesh_omega, image), grad_I(mesh_omega, image), alpha = alpha, gamma = gamma)
    u, loss , step = update_GD(mesh_omega, mesh_Omega_0, image, u, -shape_gradient, step = step * coeffStep, minStep = minStep)
    # mesh_omega = Mesh(mesh_Omega_0)
    # ALE.move(mesh_omega, u)
    u_Omega_0.vector()[:] = u.vector()[:]
    # Print and store result
    print(f"it = {k}  |  loss = {loss:.10e}    ", end = "\r")
    loss_vect.append(loss)

    # dmech.write_VTU_file(
    #     filebasename = "omega_sol",
    #     function = I(mesh_omega, image),
    #     time = k,
    #     preserve_connectivity = True)
    dmech.write_VTU_file(
    filebasename = "mapping_lung_2D",
    function = u_Omega_0,
    time = k,
    preserve_connectivity = True)
# %%
