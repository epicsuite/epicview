# state file generated using paraview version 5.13.2
import paraview
paraview.compatibility.major = 5
paraview.compatibility.minor = 13

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# grab additional command line arguments 
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--session',   type=str, default='', required=True,  help='session file to load')
namespace, extra = parser.parse_known_args()

import yaml
# load the session file
session = None
with open(namespace.session) as stream:
    try:
        session = yaml.load(stream, Loader = yaml.CLoader)
    except yaml.YAMLError as exc:
        print(exc)
sessionData = session['datasets']

# set application prefix dir 
import sys
index = sys.executable.index('Contents') + len('Contents')
appprefix = sys.executable[:index]

# load plugins
LoadPlugin(appprefix + "/Plugins/TopologyToolKit/TopologyToolKit.so", ns=globals())

# set script prefix dir 
import os
scriptprefix = os.path.dirname(sys.argv[0])
# get settings
settings = None
with open(scriptprefix + "/settings.yaml") as stream:
    try:
        settings = yaml.load(stream, Loader = yaml.CLoader)
    except yaml.YAMLError as exc:
        print(exc)

def getLabel(s, i):
    sd = s['datasets']

    return s['session']['name'] + " Chr" + str(sd['chromosome']) + " " + sd[i]['test'] + " " + str(sd['timevalues'][0]) + sd['timeunits']

# set application values
mockvtpFilename  = sessionData[0]['files'][0]
a229EvtpFilename = sessionData[1]['files'][0]

# ----------------------------------------------------------------
# setup views used in the visualization
# ----------------------------------------------------------------

# Create a new 'Light'
light1 = CreateLight()
light1.Intensity = 0.0

# get the material library
materialLibrary1 = GetMaterialLibrary()

# create light
# Create a new 'Render View'
renderView2 = CreateView('RenderView')
renderView2.ViewSize = [815, 353]
renderView2.AxesGrid = 'Grid Axes 3D Actor'
renderView2.CenterOfRotation = [-37.212897300720215, 33.836917877197266, 50.346303939819336]
renderView2.KeyLightIntensity = 1.0
renderView2.StereoType = 'Crystal Eyes'
renderView2.CameraPosition = [-50.50242769430507, 45.39805650622624, 63.05430374044836]
renderView2.CameraFocalPoint = [-37.21289730072025, 33.836917877197294, 50.34630393981924]
renderView2.CameraViewUp = [0.7115113442071805, 0.04725093970153431, 0.7010841288759954]
renderView2.CameraFocalDisk = 1.0
renderView2.CameraParallelScale = 10.041302683090546
renderView2.LegendGrid = 'Legend Grid Actor'
renderView2.PolarGrid = 'Polar Grid Actor'
renderView2.BackEnd = 'OSPRay raycaster'
renderView2.AdditionalLights = light1
renderView2.OSPRayMaterialLibrary = materialLibrary1

# view label
viewLabel2 = Text(registrationName='LeftLabel')
viewLabel2.Text = getLabel(session, 0) 
viewLabel2Display = Show(viewLabel2, renderView2, 'TextSourceRepresentation')
viewLabel2Display.Color =    settings['view']['label']['color'] 
viewLabel2Display.FontSize = settings['view']['label']['fontsize']
viewLabel2Display.Bold =     settings['view']['label']['bold']

# Create a new 'Light'
light2 = CreateLight()
light2.Intensity = 0.0

# create light
# Create a new 'Render View'
renderView3 = CreateView('RenderView')
renderView3.ViewSize = [814, 353]
renderView3.AxesGrid = 'Grid Axes 3D Actor'
renderView3.CenterOfRotation = [-37.212897300720215, 33.836917877197266, 50.346303939819336]
renderView3.KeyLightIntensity = 1.0
renderView3.StereoType = 'Crystal Eyes'
renderView3.CameraPosition = [-50.50242769430507, 45.39805650622624, 63.05430374044836]
renderView3.CameraFocalPoint = [-37.21289730072025, 33.836917877197294, 50.34630393981924]
renderView3.CameraViewUp = [0.7115113442071805, 0.04725093970153431, 0.7010841288759954]
renderView3.CameraFocalDisk = 1.0
renderView3.CameraParallelScale = 10.041302683090546
renderView3.LegendGrid = 'Legend Grid Actor'
renderView3.PolarGrid = 'Polar Grid Actor'
renderView3.BackEnd = 'OSPRay raycaster'
renderView3.AdditionalLights = light2
renderView3.OSPRayMaterialLibrary = materialLibrary1

# view label
viewLabel3 = Text(registrationName='RightLabel')
viewLabel3.Text = getLabel(session, 1) 
viewLabel3Display = Show(viewLabel3, renderView3, 'TextSourceRepresentation')
viewLabel3Display.Color =    settings['view']['label']['color'] 
viewLabel3Display.FontSize = settings['view']['label']['fontsize']
viewLabel3Display.Bold =     settings['view']['label']['bold']

SetActiveView(None)

# ----------------------------------------------------------------
# setup view layouts
# ----------------------------------------------------------------

# create new layout object 'Layout #1'
layout1 = CreateLayout(name='Layout #1')
layout1.SplitHorizontal(0, 0.500000)
layout1.AssignView(1, renderView2)
layout1.AssignView(2, renderView3)
layout1.SetSize(1630, 353)

# ----------------------------------------------------------------
# restore active view
SetActiveView(renderView2)
# ----------------------------------------------------------------

# ----------------------------------------------------------------
# setup the data processing pipelines
# ----------------------------------------------------------------

# create a new 'XML PolyData Reader'
a229Evtp = XMLPolyDataReader(registrationName='229E.vtp', FileName=[a229EvtpFilename])
a229Evtp.PointArrayStatus = ['id', 'centromere', 'compartment', 'ATAC_B', 'RAD21_A', 'CTCF_C', 'ATAC_C', 'Lb1_A', 'CTCF_A', 'RAD21_C', 'ATAC_A', 'Lb1_C', 'ATAC', 'RAD21', 'CTCF', 'Lb1', 'RAD21_log2FoldChange', 'RAD21_log10pvalue', 'CTCF_log2FoldChange', 'CTCF_log10pvalue', 'Lb1_log2FoldChange', 'Lb1_log10pvalue']
a229Evtp.TimeArray = 'None'

# create a new 'Calculator'
calculator1 = Calculator(registrationName='Calculator1', Input=a229Evtp)
calculator1.Function = ''

# create a new 'Tube'
tube4 = Tube(registrationName='Tube4', Input=calculator1)
tube4.Scalars = ['POINTS', 'RAD21']
tube4.Vectors = ['POINTS', '1']
tube4.NumberofSides = 30
tube4.Radius = 0.02
tube4.VaryRadius = 'By Absolute Scalar'

# create a new 'Delaunay 3D'
delaunay3D2 = Delaunay3D(registrationName='Delaunay3D2', Input=calculator1)

# create a new 'Gaussian Resampling'
gaussianResampling3 = GaussianResampling(registrationName='GaussianResampling3', Input=calculator1)
gaussianResampling3.ResampleField = ['POINTS', 'Lb1']
gaussianResampling3.SplatAccumulationMode = 'Sum'

# create a new 'Tube'
tube5 = Tube(registrationName='Tube5', Input=calculator1)
tube5.Scalars = ['POINTS', 'id']
tube5.Vectors = ['POINTS', '1']
tube5.NumberofSides = 50
tube5.Radius = 0.09

# create a new 'Iso Volume'
isoVolume2 = IsoVolume(registrationName='IsoVolume2', Input=tube5)
isoVolume2.InputScalars = ['POINTS', 'id']
isoVolume2.ThresholdRange = [75466232.7099999, 80085866.08]

# create a new 'XML PolyData Reader'
mockvtp = XMLPolyDataReader(registrationName='mock.vtp', FileName=[mockvtpFilename])
mockvtp.PointArrayStatus = ['id', 'centromere', 'compartment', 'ATAC_C', 'RAD21_C', 'CTCF_C', 'ATAC_B', 'Lb1_C', 'RAD21_A', 'ATAC_A', 'CTCF_A', 'Lb1_A', 'ATAC', 'RAD21', 'CTCF', 'Lb1', 'RAD21_log2FoldChange', 'RAD21_log10pvalue', 'CTCF_log2FoldChange', 'CTCF_log10pvalue', 'Lb1_log2FoldChange', 'Lb1_log10pvalue']
mockvtp.TimeArray = 'None'

# create a new 'TTK ScalarFieldSmoother'
tTKScalarFieldSmoother2 = TTKScalarFieldSmoother(registrationName='TTKScalarFieldSmoother2', Input=delaunay3D2)
tTKScalarFieldSmoother2.ScalarField = ['POINTS', 'centromere']
tTKScalarFieldSmoother2.IterationNumber = 4
tTKScalarFieldSmoother2.MaskField = ['POINTS', 'id']

# create a new 'Tube'
tube1 = Tube(registrationName='Tube1', Input=calculator1)
tube1.Scalars = ['POINTS', 'CTCF_log2FoldChangeDown']
tube1.Vectors = ['POINTS', '1']
tube1.NumberofSides = 30
tube1.Radius = 0.02
tube1.VaryRadius = 'By Absolute Scalar'

# create a new 'Tube'
tube2 = Tube(registrationName='Tube2', Input=calculator1)
tube2.Scalars = ['POINTS', 'CTCF_log2FoldChangeUp']
tube2.Vectors = ['POINTS', '1']
tube2.NumberofSides = 30
tube2.Radius = 0.02
tube2.VaryRadius = 'By Absolute Scalar'

# create a new 'Programmable Filter'
programmableFilter1 = ProgrammableFilter(registrationName='ProgrammableFilter1', Input=[mockvtp, a229Evtp])
programmableFilter1.Script = """import numpy as np
from vtk.util.numpy_support import vtk_to_numpy
import vtk

def compute_similarity_transform(A, B):
    \"\"\"
    Compute the similarity transformation (s, R, t) that best aligns each point in A
    to the corresponding point in B, including uniform scaling.
    \"\"\"
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Ensure a proper rotation by enforcing a right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute scaling
    scale = np.sum(S) / np.sum(AA ** 2) # Umeyama
    scale = (np.sum(BB ** 2) / np.sum(AA ** 2))**0.5 # wiki
    scale = 1 # no scaling
    # Compute translation
    t = centroid_B - scale * np.dot(R, centroid_A)

    return scale, R, t

def apply_similarity_transformation(points, s, R, t):
    \"\"\"
    Apply the computed similarity transformation (s, R, t) to the points.
    \"\"\"
    return s * np.dot(points, R.T) + t

def get_numpy_points_from_input(vtk_points_object):
    num_points = vtk_points_object.GetNumberOfPoints()
    return np.array([vtk_points_object.GetPoint(i) for i in range(num_points)])

# Assuming inputs[0] and inputs[1] are vtkPolyData
points0 = get_numpy_points_from_input(inputs[0])
points1 = get_numpy_points_from_input(inputs[1])

index0 = inputs[0].GetPointData().GetArray(\'id\')
index_np0 = vtk_to_numpy(index0)
index1 = inputs[1].GetPointData().GetArray(\'id\')
index_np1 = vtk_to_numpy(index1)

# Condition: Last 4 values should be zero
condition0 = (index_np0 % 250000) == 0
condition1 = (index_np1 % 250000) == 0

# Filter pointsA to keep only those points that satisfy the condition
pointsA = points0[condition0]
pointsB = points1[condition1]

print(len(pointsA), len(pointsB))
# Compute the similarity transformation that aligns points A to points B
s, R, t = compute_similarity_transform(pointsA, pointsB)

# Apply the similarity transformation to points A
transformed_points0 = apply_similarity_transformation(points0, s, R, t)

# Update the output with transformed points
output_points = vtk.vtkPoints()
for i in range(len(transformed_points0)):
    output_points.InsertNextPoint(transformed_points0[i])

output = self.GetOutput()
output.SetPoints(output_points)
"""
programmableFilter1.RequestInformationScript = ''
programmableFilter1.RequestUpdateExtentScript = ''
programmableFilter1.CopyArrays = 1
programmableFilter1.PythonPath = ''

# create a new 'Tube'
tube4_1 = Tube(registrationName='Tube4', Input=programmableFilter1)
tube4_1.Scalars = ['POINTS', 'RAD21']
tube4_1.Vectors = ['POINTS', '1']
tube4_1.NumberofSides = 30
tube4_1.Radius = 0.02
tube4_1.VaryRadius = 'By Absolute Scalar'

# create a new 'Tube'
tube1_1 = Tube(registrationName='Tube1', Input=programmableFilter1)
tube1_1.Scalars = ['POINTS', 'CTCF_log2FoldChangeDown']
tube1_1.Vectors = ['POINTS', '1']
tube1_1.NumberofSides = 20
tube1_1.Radius = 0.02
tube1_1.VaryRadius = 'By Absolute Scalar'

# create a new 'Tube'
tube2_1 = Tube(registrationName='Tube2', Input=programmableFilter1)
tube2_1.Scalars = ['POINTS', 'CTCF_log2FoldChangeUp']
tube2_1.Vectors = ['POINTS', '1']
tube2_1.NumberofSides = 30
tube2_1.Radius = 0.02
tube2_1.VaryRadius = 'By Absolute Scalar'

# create a new 'Tube'
tube3 = Tube(registrationName='Tube3', Input=programmableFilter1)
tube3.Scalars = ['POINTS', 'Lb1']
tube3.Vectors = ['POINTS', '1']
tube3.NumberofSides = 30
tube3.Radius = 0.02
tube3.VaryRadius = 'By Absolute Scalar'

# create a new 'Tube'
tube5_1 = Tube(registrationName='Tube5', Input=programmableFilter1)
tube5_1.Scalars = ['POINTS', 'id']
tube5_1.Vectors = ['POINTS', '1']
tube5_1.NumberofSides = 50
tube5_1.Radius = 0.09

# create a new 'Iso Volume'
isoVolume1 = IsoVolume(registrationName='IsoVolume1', Input=tube5_1)
isoVolume1.InputScalars = ['POINTS', 'id']
isoVolume1.ThresholdRange = [14075000.0, 14100000.0]

# create a new 'Delaunay 3D'
delaunay3D1 = Delaunay3D(registrationName='Delaunay3D1', Input=programmableFilter1)

# create a new 'TTK ScalarFieldSmoother'
tTKScalarFieldSmoother1 = TTKScalarFieldSmoother(registrationName='TTKScalarFieldSmoother1', Input=delaunay3D1)
tTKScalarFieldSmoother1.ScalarField = ['POINTS', 'centromere']
tTKScalarFieldSmoother1.IterationNumber = 4
tTKScalarFieldSmoother1.MaskField = ['POINTS', 'id']

# create a new 'Contour'
contour8 = Contour(registrationName='Contour8', Input=tTKScalarFieldSmoother1)
contour8.ContourBy = ['POINTS', 'centromere']
contour8.Isosurfaces = [0.5]
contour8.PointMergeMethod = 'Uniform Binning'

# create a new 'TTK GeometrySmoother'
tTKGeometrySmoother4 = TTKGeometrySmoother(registrationName='TTKGeometrySmoother4', Input=contour8)
tTKGeometrySmoother4.IterationNumber = 3
tTKGeometrySmoother4.InputMaskField = ['POINTS', 'compartment']

# create a new 'Iso Volume'
isoVolume3 = IsoVolume(registrationName='IsoVolume3', Input=delaunay3D1)
isoVolume3.InputScalars = ['POINTS', 'compartment']
isoVolume3.ThresholdRange = [-1.0, -0.04]

# create a new 'Gaussian Resampling'
gaussianResampling6 = GaussianResampling(registrationName='GaussianResampling6', Input=isoVolume3)
gaussianResampling6.ResampleField = ['POINTS', 'ignore arrays']
gaussianResampling6.GaussianExponentFactor = -0.1
gaussianResampling6.ScaleSplats = 0
gaussianResampling6.EllipticalSplats = 0

# create a new 'Gaussian Resampling'
gaussianResampling2 = GaussianResampling(registrationName='GaussianResampling2', Input=calculator1)
gaussianResampling2.ResampleField = ['POINTS', 'RAD21']
gaussianResampling2.SplatAccumulationMode = 'Sum'

# create a new 'Contour'
contour2 = Contour(registrationName='Contour2', Input=gaussianResampling2)
contour2.ContourBy = ['POINTS', 'SplatterValues']
contour2.Isosurfaces = [1.0]
contour2.PointMergeMethod = 'Uniform Binning'

# create a new 'Tube'
tube3_1 = Tube(registrationName='Tube3', Input=calculator1)
tube3_1.Scalars = ['POINTS', 'Lb1']
tube3_1.Vectors = ['POINTS', '1']
tube3_1.NumberofSides = 30
tube3_1.Radius = 0.02
tube3_1.VaryRadius = 'By Absolute Scalar'

# create a new 'Gaussian Resampling'
gaussianResampling3_1 = GaussianResampling(registrationName='GaussianResampling3', Input=programmableFilter1)
gaussianResampling3_1.ResampleField = ['POINTS', 'RAD21']
gaussianResampling3_1.SplatAccumulationMode = 'Sum'

# create a new 'Contour'
contour3 = Contour(registrationName='Contour3', Input=gaussianResampling3_1)
contour3.ContourBy = ['POINTS', 'SplatterValues']
contour3.Isosurfaces = [1.0]
contour3.PointMergeMethod = 'Uniform Binning'

# create a new 'Contour'
contour6 = Contour(registrationName='Contour6', Input=tTKScalarFieldSmoother2)
contour6.ContourBy = ['POINTS', 'centromere']
contour6.Isosurfaces = [-0.5]
contour6.PointMergeMethod = 'Uniform Binning'

# create a new 'TTK GeometrySmoother'
tTKGeometrySmoother2 = TTKGeometrySmoother(registrationName='TTKGeometrySmoother2', Input=contour6)
tTKGeometrySmoother2.IterationNumber = 3
tTKGeometrySmoother2.InputMaskField = ['POINTS', 'compartment']

# create a new 'Contour'
contour4 = Contour(registrationName='Contour4', Input=tTKScalarFieldSmoother2)
contour4.ContourBy = ['POINTS', 'centromere']
contour4.Isosurfaces = [0.5]
contour4.PointMergeMethod = 'Uniform Binning'

# create a new 'TTK GeometrySmoother'
tTKGeometrySmoother3 = TTKGeometrySmoother(registrationName='TTKGeometrySmoother3', Input=contour4)
tTKGeometrySmoother3.IterationNumber = 3
tTKGeometrySmoother3.InputMaskField = ['POINTS', 'centromere']

# create a new 'Contour'
contour5 = Contour(registrationName='Contour5', Input=tTKScalarFieldSmoother1)
contour5.ContourBy = ['POINTS', 'centromere']
contour5.Isosurfaces = [-0.5]
contour5.PointMergeMethod = 'Uniform Binning'

# create a new 'TTK GeometrySmoother'
tTKGeometrySmoother1 = TTKGeometrySmoother(registrationName='TTKGeometrySmoother1', Input=contour5)
tTKGeometrySmoother1.IterationNumber = 3
tTKGeometrySmoother1.InputMaskField = ['POINTS', 'compartment']

# create a new 'Contour'
contour7 = Contour(registrationName='Contour7', Input=gaussianResampling6)
contour7.ContourBy = ['POINTS', 'SplatterValues']
contour7.Isosurfaces = [0.99]
contour7.PointMergeMethod = 'Uniform Binning'

# create a new 'Gaussian Resampling'
gaussianResampling2_1 = GaussianResampling(registrationName='GaussianResampling2', Input=programmableFilter1)
gaussianResampling2_1.ResampleField = ['POINTS', 'Lb1']
gaussianResampling2_1.SplatAccumulationMode = 'Sum'

# create a new 'Contour'
contour2_1 = Contour(registrationName='Contour2', Input=gaussianResampling2_1)
contour2_1.ContourBy = ['POINTS', 'SplatterValues']
contour2_1.Isosurfaces = [1.0]
contour2_1.PointMergeMethod = 'Uniform Binning'

# create a new 'Gaussian Resampling'
gaussianResampling1 = GaussianResampling(registrationName='GaussianResampling1', Input=programmableFilter1)
gaussianResampling1.ResampleField = ['POINTS', 'CTCF']
gaussianResampling1.SplatAccumulationMode = 'Sum'

# create a new 'Contour'
contour1 = Contour(registrationName='Contour1', Input=gaussianResampling1)
contour1.ContourBy = ['POINTS', 'SplatterValues']
contour1.Isosurfaces = [1.0]
contour1.PointMergeMethod = 'Uniform Binning'

# create a new 'Contour'
contour3_1 = Contour(registrationName='Contour3', Input=gaussianResampling3)
contour3_1.ContourBy = ['POINTS', 'SplatterValues']
contour3_1.Isosurfaces = [1.0]
contour3_1.PointMergeMethod = 'Uniform Binning'

# create a new 'Gaussian Resampling'
gaussianResampling1_1 = GaussianResampling(registrationName='GaussianResampling1', Input=calculator1)
gaussianResampling1_1.ResampleField = ['POINTS', 'CTCF']
gaussianResampling1_1.SplatAccumulationMode = 'Sum'

# create a new 'Contour'
contour1_1 = Contour(registrationName='Contour1', Input=gaussianResampling1_1)
contour1_1.ContourBy = ['POINTS', 'SplatterValues']
contour1_1.Isosurfaces = [1.0]
contour1_1.PointMergeMethod = 'Uniform Binning'

# ----------------------------------------------------------------
# setup the visualization in view 'renderView2'
# ----------------------------------------------------------------

# show data from tube5_1
tube5_1Display = Show(tube5_1, renderView2, 'GeometryRepresentation')

# get 2D transfer function for 'compartment'
compartmentTF2D = GetTransferFunction2D('compartment')
compartmentTF2D.ScalarRangeInitialized = 1
compartmentTF2D.Range = [-0.04, 0.04, 0.0, 1.0]

# get color transfer function/color map for 'compartment'
compartmentLUT = GetColorTransferFunction('compartment')
compartmentLUT.TransferFunction2D = compartmentTF2D
compartmentLUT.RGBPoints = [-0.04, 0.0, 0.266667, 0.105882, -0.03529412000000001, 0.062284, 0.386621, 0.170473, -0.030274519999999985, 0.15917, 0.516263, 0.251211, -0.025254919999999986, 0.314187, 0.649135, 0.354556, -0.020235280000000008, 0.493195, 0.765398, 0.496655, -0.015215680000000002, 0.670588, 0.866897, 0.647059, -0.010196080000000003, 0.796078, 0.91857, 0.772549, -0.005176480000000011, 0.892503, 0.950865, 0.877278, -0.00015686279999999053, 0.966321, 0.968089, 0.965859, 0.004862760000000001, 0.930488, 0.885198, 0.932872, 0.009882360000000007, 0.871742, 0.788005, 0.886736, 0.014901959999999999, 0.7807, 0.672357, 0.825221, 0.019921559999999998, 0.681968, 0.545175, 0.742561, 0.02494116000000001, 0.583852, 0.40692, 0.652134, 0.029960800000000017, 0.497732, 0.234679, 0.55371, 0.034980400000000016, 0.383852, 0.103345, 0.431911, 0.04, 0.25098, 0.0, 0.294118]
compartmentLUT.ColorSpace = 'Lab'
compartmentLUT.NanColor = [0.25, 0.0, 0.0]
compartmentLUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
tube5_1Display.Representation = 'Surface'
tube5_1Display.AmbientColor = [0.39215686274509803, 0.39215686274509803, 0.39215686274509803]
tube5_1Display.ColorArrayName = ['POINTS', 'compartment']
tube5_1Display.DiffuseColor = [0.39215686274509803, 0.39215686274509803, 0.39215686274509803]
tube5_1Display.LookupTable = compartmentLUT
tube5_1Display.Specular = 1.0
tube5_1Display.SpecularPower = 50.0
tube5_1Display.SelectNormalArray = 'TubeNormals'
tube5_1Display.SelectTangentArray = 'None'
tube5_1Display.SelectTCoordArray = 'None'
tube5_1Display.TextureTransform = 'Transform2'
tube5_1Display.OSPRayScaleArray = 'id'
tube5_1Display.OSPRayScaleFunction = 'Piecewise Function'
tube5_1Display.Assembly = ''
tube5_1Display.SelectedBlockSelectors = ['']
tube5_1Display.SelectOrientationVectors = 'None'
tube5_1Display.ScaleFactor = 0.5700534224510193
tube5_1Display.SelectScaleArray = 'id'
tube5_1Display.GlyphType = 'Arrow'
tube5_1Display.GlyphTableIndexArray = 'id'
tube5_1Display.GaussianRadius = 0.028502671122550966
tube5_1Display.SetScaleArray = ['POINTS', 'id']
tube5_1Display.ScaleTransferFunction = 'Piecewise Function'
tube5_1Display.OpacityArray = ['POINTS', 'id']
tube5_1Display.OpacityTransferFunction = 'Piecewise Function'
tube5_1Display.DataAxesGrid = 'Grid Axes Representation'
tube5_1Display.PolarAxes = 'Polar Axes Representation'
tube5_1Display.SelectInputVectors = ['POINTS', 'TubeNormals']
tube5_1Display.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
tube5_1Display.ScaleTransferFunction.Points = [4.0, 0.0, 0.5, 0.0, 134750000.0, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
tube5_1Display.OpacityTransferFunction.Points = [4.0, 0.0, 0.5, 0.0, 134750000.0, 1.0, 0.5, 0.0]

# init the 'Polar Axes Representation' selected for 'PolarAxes'
tube5_1Display.PolarAxes.EnableOverallColor = 0
tube5_1Display.PolarAxes.DeltaRangeMajor = 10.0
tube5_1Display.PolarAxes.DeltaRangeMinor = 5.0
tube5_1Display.PolarAxes.ArcTickMatchesRadialAxes = 0

# show data from delaunay3D1
delaunay3D1Display = Show(delaunay3D1, renderView2, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
delaunay3D1Display.Representation = 'Surface'
delaunay3D1Display.ColorArrayName = ['POINTS', '']
delaunay3D1Display.Opacity = 0.05
delaunay3D1Display.SelectNormalArray = 'None'
delaunay3D1Display.SelectTangentArray = 'None'
delaunay3D1Display.SelectTCoordArray = 'None'
delaunay3D1Display.TextureTransform = 'Transform2'
delaunay3D1Display.OSPRayScaleArray = 'id'
delaunay3D1Display.OSPRayScaleFunction = 'Piecewise Function'
delaunay3D1Display.Assembly = ''
delaunay3D1Display.SelectedBlockSelectors = ['']
delaunay3D1Display.SelectOrientationVectors = 'None'
delaunay3D1Display.ScaleFactor = 1.0989726066589356
delaunay3D1Display.SelectScaleArray = 'id'
delaunay3D1Display.GlyphType = 'Arrow'
delaunay3D1Display.GlyphTableIndexArray = 'id'
delaunay3D1Display.GaussianRadius = 0.05494863033294678
delaunay3D1Display.SetScaleArray = ['POINTS', 'id']
delaunay3D1Display.ScaleTransferFunction = 'Piecewise Function'
delaunay3D1Display.OpacityArray = ['POINTS', 'id']
delaunay3D1Display.OpacityTransferFunction = 'Piecewise Function'
delaunay3D1Display.DataAxesGrid = 'Grid Axes Representation'
delaunay3D1Display.PolarAxes = 'Polar Axes Representation'
delaunay3D1Display.ScalarOpacityUnitDistance = 0.3607313511815122
delaunay3D1Display.OpacityArrayName = ['POINTS', 'id']
delaunay3D1Display.SelectInputVectors = [None, '']
delaunay3D1Display.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
delaunay3D1Display.ScaleTransferFunction.Points = [50000.0, 0.0, 0.5, 0.0, 45100000.0, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
delaunay3D1Display.OpacityTransferFunction.Points = [50000.0, 0.0, 0.5, 0.0, 45100000.0, 1.0, 0.5, 0.0]

# show data from tTKGeometrySmoother1
tTKGeometrySmoother1Display = Show(tTKGeometrySmoother1, renderView2, 'GeometryRepresentation')

# get 2D transfer function for 'centromere'
centromereTF2D = GetTransferFunction2D('centromere')
centromereTF2D.ScalarRangeInitialized = 1
centromereTF2D.Range = [-0.1, 0.1, 0.0, 1.0]

# get color transfer function/color map for 'centromere'
centromereLUT = GetColorTransferFunction('centromere')
centromereLUT.TransferFunction2D = centromereTF2D
centromereLUT.RGBPoints = [-0.5, 0.23137254902, 0.298039215686, 0.752941176471, 6.103515625e-05, 0.865, 0.865, 0.865, 0.5001220703125, 0.705882352941, 0.0156862745098, 0.149019607843]
centromereLUT.ScalarRangeInitialized = 1.0

# trace defaults for the display properties.
tTKGeometrySmoother1Display.Representation = 'Surface'
tTKGeometrySmoother1Display.ColorArrayName = ['POINTS', 'centromere']
tTKGeometrySmoother1Display.LookupTable = centromereLUT
tTKGeometrySmoother1Display.Opacity = 0.8
tTKGeometrySmoother1Display.SelectNormalArray = 'Normals'
tTKGeometrySmoother1Display.SelectTangentArray = 'None'
tTKGeometrySmoother1Display.SelectTCoordArray = 'None'
tTKGeometrySmoother1Display.TextureTransform = 'Transform2'
tTKGeometrySmoother1Display.OSPRayScaleArray = 'compartment'
tTKGeometrySmoother1Display.OSPRayScaleFunction = 'Piecewise Function'
tTKGeometrySmoother1Display.Assembly = ''
tTKGeometrySmoother1Display.SelectedBlockSelectors = ['']
tTKGeometrySmoother1Display.SelectOrientationVectors = 'None'
tTKGeometrySmoother1Display.ScaleFactor = 0.41825361251831056
tTKGeometrySmoother1Display.SelectScaleArray = 'compartment'
tTKGeometrySmoother1Display.GlyphType = 'Arrow'
tTKGeometrySmoother1Display.GlyphTableIndexArray = 'compartment'
tTKGeometrySmoother1Display.GaussianRadius = 0.020912680625915527
tTKGeometrySmoother1Display.SetScaleArray = ['POINTS', 'compartment']
tTKGeometrySmoother1Display.ScaleTransferFunction = 'Piecewise Function'
tTKGeometrySmoother1Display.OpacityArray = ['POINTS', 'compartment']
tTKGeometrySmoother1Display.OpacityTransferFunction = 'Piecewise Function'
tTKGeometrySmoother1Display.DataAxesGrid = 'Grid Axes Representation'
tTKGeometrySmoother1Display.PolarAxes = 'Polar Axes Representation'
tTKGeometrySmoother1Display.SelectInputVectors = ['POINTS', 'Normals']
tTKGeometrySmoother1Display.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
tTKGeometrySmoother1Display.ScaleTransferFunction.Points = [-0.05, 0.0, 0.5, 0.0, -0.04999237135052681, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
tTKGeometrySmoother1Display.OpacityTransferFunction.Points = [-0.05, 0.0, 0.5, 0.0, -0.04999237135052681, 1.0, 0.5, 0.0]

# show data from tTKGeometrySmoother4
tTKGeometrySmoother4Display = Show(tTKGeometrySmoother4, renderView2, 'GeometryRepresentation')

# trace defaults for the display properties.
tTKGeometrySmoother4Display.Representation = 'Surface'
tTKGeometrySmoother4Display.ColorArrayName = ['POINTS', 'centromere']
tTKGeometrySmoother4Display.LookupTable = centromereLUT
tTKGeometrySmoother4Display.Opacity = 0.8
tTKGeometrySmoother4Display.SelectNormalArray = 'Normals'
tTKGeometrySmoother4Display.SelectTangentArray = 'None'
tTKGeometrySmoother4Display.SelectTCoordArray = 'None'
tTKGeometrySmoother4Display.TextureTransform = 'Transform2'
tTKGeometrySmoother4Display.OSPRayScaleArray = 'centromere'
tTKGeometrySmoother4Display.OSPRayScaleFunction = 'Piecewise Function'
tTKGeometrySmoother4Display.Assembly = ''
tTKGeometrySmoother4Display.SelectedBlockSelectors = ['']
tTKGeometrySmoother4Display.SelectOrientationVectors = 'None'
tTKGeometrySmoother4Display.ScaleFactor = 0.652145004272461
tTKGeometrySmoother4Display.SelectScaleArray = 'centromere'
tTKGeometrySmoother4Display.GlyphType = 'Arrow'
tTKGeometrySmoother4Display.GlyphTableIndexArray = 'centromere'
tTKGeometrySmoother4Display.GaussianRadius = 0.03260725021362305
tTKGeometrySmoother4Display.SetScaleArray = ['POINTS', 'centromere']
tTKGeometrySmoother4Display.ScaleTransferFunction = 'Piecewise Function'
tTKGeometrySmoother4Display.OpacityArray = ['POINTS', 'centromere']
tTKGeometrySmoother4Display.OpacityTransferFunction = 'Piecewise Function'
tTKGeometrySmoother4Display.DataAxesGrid = 'Grid Axes Representation'
tTKGeometrySmoother4Display.PolarAxes = 'Polar Axes Representation'
tTKGeometrySmoother4Display.SelectInputVectors = ['POINTS', 'Normals']
tTKGeometrySmoother4Display.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
tTKGeometrySmoother4Display.ScaleTransferFunction.Points = [-0.5, 0.0, 0.5, 0.0, -0.49993896484375, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
tTKGeometrySmoother4Display.OpacityTransferFunction.Points = [-0.5, 0.0, 0.5, 0.0, -0.49993896484375, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for centromereLUT in view renderView2
centromereLUTColorBar = GetScalarBar(centromereLUT, renderView2)
centromereLUTColorBar.Position = [0.8199999999999998, 0.01]
centromereLUTColorBar.Title = 'centromere'
centromereLUTColorBar.ComponentTitle = ''

# set color bar visibility
centromereLUTColorBar.Visibility = 1

# get color legend/bar for compartmentLUT in view renderView2
compartmentLUTColorBar = GetScalarBar(compartmentLUT, renderView2)
compartmentLUTColorBar.WindowLocation = 'Upper Right Corner'
compartmentLUTColorBar.Position = [0.8246715610510046, 0.6559139784946236]
compartmentLUTColorBar.Title = 'compartment'
compartmentLUTColorBar.ComponentTitle = ''

# set color bar visibility
compartmentLUTColorBar.Visibility = 1

# show color legend
tube5_1Display.SetScalarBarVisibility(renderView2, True)

# show color legend
tTKGeometrySmoother1Display.SetScalarBarVisibility(renderView2, True)

# show color legend
tTKGeometrySmoother4Display.SetScalarBarVisibility(renderView2, True)

# ----------------------------------------------------------------
# setup the visualization in view 'renderView3'
# ----------------------------------------------------------------

# show data from tube5
tube5Display = Show(tube5, renderView3, 'GeometryRepresentation')

# trace defaults for the display properties.
tube5Display.Representation = 'Surface'
tube5Display.AmbientColor = [0.39215686274509803, 0.39215686274509803, 0.39215686274509803]
tube5Display.ColorArrayName = ['POINTS', 'compartment']
tube5Display.DiffuseColor = [0.39215686274509803, 0.39215686274509803, 0.39215686274509803]
tube5Display.LookupTable = compartmentLUT
tube5Display.Specular = 1.0
tube5Display.SpecularPower = 50.0
tube5Display.SelectNormalArray = 'TubeNormals'
tube5Display.SelectTangentArray = 'None'
tube5Display.SelectTCoordArray = 'None'
tube5Display.TextureTransform = 'Transform2'
tube5Display.OSPRayScaleArray = 'id'
tube5Display.OSPRayScaleFunction = 'Piecewise Function'
tube5Display.Assembly = ''
tube5Display.SelectedBlockSelectors = ['']
tube5Display.SelectOrientationVectors = 'None'
tube5Display.ScaleFactor = 0.4780649186395258
tube5Display.SelectScaleArray = 'id'
tube5Display.GlyphType = 'Arrow'
tube5Display.GlyphTableIndexArray = 'id'
tube5Display.GaussianRadius = 0.02390324593197629
tube5Display.SetScaleArray = ['POINTS', 'id']
tube5Display.ScaleTransferFunction = 'Piecewise Function'
tube5Display.OpacityArray = ['POINTS', 'id']
tube5Display.OpacityTransferFunction = 'Piecewise Function'
tube5Display.DataAxesGrid = 'Grid Axes Representation'
tube5Display.PolarAxes = 'Polar Axes Representation'
tube5Display.SelectInputVectors = ['POINTS', 'TubeNormals']
tube5Display.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
tube5Display.ScaleTransferFunction.Points = [63.0, 0.0, 0.5, 0.0, 134750000.0, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
tube5Display.OpacityTransferFunction.Points = [63.0, 0.0, 0.5, 0.0, 134750000.0, 1.0, 0.5, 0.0]

# init the 'Polar Axes Representation' selected for 'PolarAxes'
tube5Display.PolarAxes.EnableOverallColor = 0
tube5Display.PolarAxes.DeltaRangeMajor = 10.0
tube5Display.PolarAxes.DeltaRangeMinor = 5.0
tube5Display.PolarAxes.ArcTickMatchesRadialAxes = 0

# show data from delaunay3D2
delaunay3D2Display = Show(delaunay3D2, renderView3, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
delaunay3D2Display.Representation = 'Surface'
delaunay3D2Display.ColorArrayName = ['POINTS', '']
delaunay3D2Display.Opacity = 0.05
delaunay3D2Display.SelectNormalArray = 'None'
delaunay3D2Display.SelectTangentArray = 'None'
delaunay3D2Display.SelectTCoordArray = 'None'
delaunay3D2Display.TextureTransform = 'Transform2'
delaunay3D2Display.OSPRayScaleArray = 'id'
delaunay3D2Display.OSPRayScaleFunction = 'Piecewise Function'
delaunay3D2Display.Assembly = ''
delaunay3D2Display.SelectedBlockSelectors = ['']
delaunay3D2Display.SelectOrientationVectors = 'None'
delaunay3D2Display.ScaleFactor = 1.0581211892142386
delaunay3D2Display.SelectScaleArray = 'id'
delaunay3D2Display.GlyphType = 'Arrow'
delaunay3D2Display.GlyphTableIndexArray = 'id'
delaunay3D2Display.GaussianRadius = 0.05290605946071193
delaunay3D2Display.SetScaleArray = ['POINTS', 'id']
delaunay3D2Display.ScaleTransferFunction = 'Piecewise Function'
delaunay3D2Display.OpacityArray = ['POINTS', 'id']
delaunay3D2Display.OpacityTransferFunction = 'Piecewise Function'
delaunay3D2Display.DataAxesGrid = 'Grid Axes Representation'
delaunay3D2Display.PolarAxes = 'Polar Axes Representation'
delaunay3D2Display.ScalarOpacityUnitDistance = 0.3444155306321857
delaunay3D2Display.OpacityArrayName = ['POINTS', 'id']
delaunay3D2Display.SelectInputVectors = [None, '']
delaunay3D2Display.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
delaunay3D2Display.ScaleTransferFunction.Points = [50000.0, 0.0, 0.5, 0.0, 45100000.0, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
delaunay3D2Display.OpacityTransferFunction.Points = [50000.0, 0.0, 0.5, 0.0, 45100000.0, 1.0, 0.5, 0.0]

# show data from tTKGeometrySmoother2
tTKGeometrySmoother2Display = Show(tTKGeometrySmoother2, renderView3, 'GeometryRepresentation')

# trace defaults for the display properties.
tTKGeometrySmoother2Display.Representation = 'Surface'
tTKGeometrySmoother2Display.AmbientColor = [0.0, 0.3333333333333333, 1.0]
tTKGeometrySmoother2Display.ColorArrayName = ['POINTS', 'centromere']
tTKGeometrySmoother2Display.DiffuseColor = [0.0, 0.3333333333333333, 1.0]
tTKGeometrySmoother2Display.LookupTable = centromereLUT
tTKGeometrySmoother2Display.Opacity = 0.8
tTKGeometrySmoother2Display.SelectNormalArray = 'Normals'
tTKGeometrySmoother2Display.SelectTangentArray = 'None'
tTKGeometrySmoother2Display.SelectTCoordArray = 'None'
tTKGeometrySmoother2Display.TextureTransform = 'Transform2'
tTKGeometrySmoother2Display.OSPRayScaleArray = 'compartment'
tTKGeometrySmoother2Display.OSPRayScaleFunction = 'Piecewise Function'
tTKGeometrySmoother2Display.Assembly = ''
tTKGeometrySmoother2Display.SelectedBlockSelectors = ['']
tTKGeometrySmoother2Display.SelectOrientationVectors = 'None'
tTKGeometrySmoother2Display.ScaleFactor = 0.43479795931363974
tTKGeometrySmoother2Display.SelectScaleArray = 'compartment'
tTKGeometrySmoother2Display.GlyphType = 'Arrow'
tTKGeometrySmoother2Display.GlyphTableIndexArray = 'compartment'
tTKGeometrySmoother2Display.GaussianRadius = 0.021739897965681987
tTKGeometrySmoother2Display.SetScaleArray = ['POINTS', 'compartment']
tTKGeometrySmoother2Display.ScaleTransferFunction = 'Piecewise Function'
tTKGeometrySmoother2Display.OpacityArray = ['POINTS', 'compartment']
tTKGeometrySmoother2Display.OpacityTransferFunction = 'Piecewise Function'
tTKGeometrySmoother2Display.DataAxesGrid = 'Grid Axes Representation'
tTKGeometrySmoother2Display.PolarAxes = 'Polar Axes Representation'
tTKGeometrySmoother2Display.SelectInputVectors = ['POINTS', 'Normals']
tTKGeometrySmoother2Display.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
tTKGeometrySmoother2Display.ScaleTransferFunction.Points = [-0.05, 0.0, 0.5, 0.0, -0.04999237135052681, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
tTKGeometrySmoother2Display.OpacityTransferFunction.Points = [-0.05, 0.0, 0.5, 0.0, -0.04999237135052681, 1.0, 0.5, 0.0]

# show data from tTKGeometrySmoother3
tTKGeometrySmoother3Display = Show(tTKGeometrySmoother3, renderView3, 'GeometryRepresentation')

# trace defaults for the display properties.
tTKGeometrySmoother3Display.Representation = 'Surface'
tTKGeometrySmoother3Display.AmbientColor = [1.0, 0.0, 0.0]
tTKGeometrySmoother3Display.ColorArrayName = ['POINTS', 'centromere']
tTKGeometrySmoother3Display.DiffuseColor = [1.0, 0.0, 0.0]
tTKGeometrySmoother3Display.LookupTable = centromereLUT
tTKGeometrySmoother3Display.Opacity = 0.8
tTKGeometrySmoother3Display.SelectNormalArray = 'Normals'
tTKGeometrySmoother3Display.SelectTangentArray = 'None'
tTKGeometrySmoother3Display.SelectTCoordArray = 'None'
tTKGeometrySmoother3Display.TextureTransform = 'Transform2'
tTKGeometrySmoother3Display.OSPRayScaleArray = 'centromere'
tTKGeometrySmoother3Display.OSPRayScaleFunction = 'Piecewise Function'
tTKGeometrySmoother3Display.Assembly = ''
tTKGeometrySmoother3Display.SelectedBlockSelectors = ['']
tTKGeometrySmoother3Display.SelectOrientationVectors = 'None'
tTKGeometrySmoother3Display.ScaleFactor = 0.792158910246495
tTKGeometrySmoother3Display.SelectScaleArray = 'centromere'
tTKGeometrySmoother3Display.GlyphType = 'Arrow'
tTKGeometrySmoother3Display.GlyphTableIndexArray = 'centromere'
tTKGeometrySmoother3Display.GaussianRadius = 0.03960794551232475
tTKGeometrySmoother3Display.SetScaleArray = ['POINTS', 'centromere']
tTKGeometrySmoother3Display.ScaleTransferFunction = 'Piecewise Function'
tTKGeometrySmoother3Display.OpacityArray = ['POINTS', 'centromere']
tTKGeometrySmoother3Display.OpacityTransferFunction = 'Piecewise Function'
tTKGeometrySmoother3Display.DataAxesGrid = 'Grid Axes Representation'
tTKGeometrySmoother3Display.PolarAxes = 'Polar Axes Representation'
tTKGeometrySmoother3Display.SelectInputVectors = ['POINTS', 'Normals']
tTKGeometrySmoother3Display.WriteLog = ''

# init the 'Piecewise Function' selected for 'ScaleTransferFunction'
tTKGeometrySmoother3Display.ScaleTransferFunction.Points = [0.05, 0.0, 0.5, 0.0, 0.05000763013958931, 1.0, 0.5, 0.0]

# init the 'Piecewise Function' selected for 'OpacityTransferFunction'
tTKGeometrySmoother3Display.OpacityTransferFunction.Points = [0.05, 0.0, 0.5, 0.0, 0.05000763013958931, 1.0, 0.5, 0.0]

# setup the color legend parameters for each legend in this view

# get color legend/bar for compartmentLUT in view renderView3
compartmentLUTColorBar_1 = GetScalarBar(compartmentLUT, renderView3)
compartmentLUTColorBar_1.WindowLocation = 'Upper Right Corner'
compartmentLUTColorBar_1.Title = 'compartment'
compartmentLUTColorBar_1.ComponentTitle = ''

# set color bar visibility
compartmentLUTColorBar_1.Visibility = 1

# get color legend/bar for centromereLUT in view renderView3
centromereLUTColorBar_1 = GetScalarBar(centromereLUT, renderView3)
centromereLUTColorBar_1.Title = 'centromere'
centromereLUTColorBar_1.ComponentTitle = ''

# set color bar visibility
centromereLUTColorBar_1.Visibility = 1

# show color legend
tube5Display.SetScalarBarVisibility(renderView3, True)

# show color legend
tTKGeometrySmoother2Display.SetScalarBarVisibility(renderView3, True)

# show color legend
tTKGeometrySmoother3Display.SetScalarBarVisibility(renderView3, True)

# ----------------------------------------------------------------
# setup color maps and opacity maps used in the visualization
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get opacity transfer function/opacity map for 'centromere'
centromerePWF = GetOpacityTransferFunction('centromere')
centromerePWF.Points = [-0.5, 0.0, 0.5, 0.0, 0.5001220703125, 1.0, 0.5, 0.0]
centromerePWF.ScalarRangeInitialized = 1

# get opacity transfer function/opacity map for 'compartment'
compartmentPWF = GetOpacityTransferFunction('compartment')
compartmentPWF.Points = [-0.04, 0.0, 0.5, 0.0, 0.04, 1.0, 0.5, 0.0]
compartmentPWF.ScalarRangeInitialized = 1

# ----------------------------------------------------------------
# setup animation scene, tracks and keyframes
# note: the Get..() functions create a new object, if needed
# ----------------------------------------------------------------

# get time animation track
timeAnimationCue1 = GetTimeTrack()

# initialize the animation scene

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# initialize the timekeeper

# initialize the animation track

# get animation scene
animationScene1 = GetAnimationScene()

# initialize the animation scene
animationScene1.ViewModules = [renderView2, renderView3]
animationScene1.Cues = timeAnimationCue1
animationScene1.AnimationTime = 0.22241992882562278

# ----------------------------------------------------------------
# restore active source
SetActiveSource(None)
# ----------------------------------------------------------------


##--------------------------------------------
## You may need to add some code at the end of this python script depending on your usage, eg:
#
## Render all views to see them appears
# RenderAllViews()
#
## Interact with the view, usefull when running from pvpython
# Interact()
#
## Save a screenshot of the active view
# SaveScreenshot("path/to/screenshot.png")
#
## Save a screenshot of a layout (multiple splitted view)
# SaveScreenshot("path/to/screenshot.png", GetLayout())
#
## Save all "Extractors" from the pipeline browser
# SaveExtracts()
#
## Save a animation of the current active view
# SaveAnimation()
#
## Please refer to the documentation of paraview.simple
## https://www.paraview.org/paraview-docs/latest/python/paraview.simple.html
##--------------------------------------------

import ptc
app = ptc.Viewer([renderView2, renderView3], from_state=True)
app.start()

