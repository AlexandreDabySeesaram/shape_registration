import vtk
import numpy as np
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk


def sign_masking_binary(
    input_name          : str   = None              , 
    suffix              : str   = "signed"          , 
    field_name          : str   = "pixel intensity" ,
    scalar2zero         : int   = 200               ,
    scalar_background   : int   = 0                 ,                                           # Initial background pixel intensity
    scalar_foreground   : int   = 100               ,                                           # Initial foreground pixel intensity
    target_value_bg     : float = 50                ,                                           # target background pixel intensity
    target_value_fg     : float = -50               ,                                           # target foreground pixel intensity
    target_type         : str   = "signed_char"     ,                                           # unsigned_char, signed_char, float
    ):                                         


    input_file      = input_name+".vti"

    # Load the VTI file
    reader          = vtk.vtkXMLImageDataReader()
    reader.SetFileName(input_file)
    reader.Update()

    # Get the image data
    image_data      = reader.GetOutput()

    # Extract scalar data and convert to Numpy array
    scalars         = image_data.GetPointData().GetScalars()
    scalar_array    = vtk_to_numpy(scalars)

    scalar_array[(scalar_array == scalar2zero)]         = 0
    scalar_array[(scalar_array == scalar_background)]   = target_value_bg
    scalar_array[(scalar_array == scalar_foreground)]   = target_value_fg
    match target_type:
        case "unsigned_char":
            scalar_array = scalar_array.astype(np.uint8)
        case "signed_char":
            scalar_array = scalar_array.astype(np.int8)
        case "float":
            scalar_array = scalar_array.astype(np.float64)

    modified_scalars = numpy_to_vtk(scalar_array)
    modified_scalars.SetName(field_name)  # Set the name of the scalar field
    image_data.GetPointData().SetScalars(modified_scalars)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(input_name+"_"+suffix+".vti")
    writer.SetInputData(image_data)
    writer.Write()

    print("Done sign masking. "+input_name)