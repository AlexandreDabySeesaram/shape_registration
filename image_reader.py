import matplotlib.pyplot as plt
from dolfin import *
import dolfin_mech as dmech
import numpy as np

img_raw = plt.imread("lung.pgm")
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
img_interpolated = interpolate(y, V)


dmech.write_VTU_file(
    filebasename = "image2",
    function = img_interpolated,
    time = 0,
    preserve_connectivity = True)