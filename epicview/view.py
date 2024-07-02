from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vtk, vuetify

from vtk.util import numpy_support
from vtkmodules.vtkFiltersGeneral import vtkTableToPolyData
from vtkmodules.vtkIOInfovis import vtkDelimitedTextReader
from vtkmodules.vtkFiltersSources import vtkSphereSource
from vtkmodules.vtkFiltersCore import (
    vtkTubeFilter,
    vtkGlyph3D,
)
from vtkmodules.vtkCommonDataModel import (
    vtkDataObject,
    vtkPointData,
    vtkPolyData,
    vtkCellArray,
)
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkPolyDataMapper,
    vtkGlyph3DMapper,
    vtkRenderer,
    vtkRenderWindow, 
    vtkRenderWindowInteractor,
)
from vtkmodules.vtkRenderingAnnotation import (
    vtkCubeAxesActor,
    vtkScalarBarActor,
)
from vtkmodules.vtkInteractionWidgets import vtkScalarBarWidget

# Required for interactor initialization
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleSwitch  # noqa

# Required for rendering initialization, not necessary for
# local rendering, but doesn't hurt to include it
import vtkmodules.vtkRenderingOpenGL2  # noqa

import pkg_resources
import os

def getModulePath():
    return os.path.dirname(pkg_resources.resource_filename(__name__, 'view.py'))

class etkServer:

    def __init__(self):
        self.ID = 0

    def start(self, structurefile, variable):
        # -----------------------------------------------------------------------------
        # VTK pipeline
        # -----------------------------------------------------------------------------

        renderer = vtkRenderer()
        renderer.SetBackground(0.5, 0.5, 0.5)
        renderWindow = vtkRenderWindow()
        renderWindow.AddRenderer(renderer)

        renderWindowInteractor = vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)
        renderWindowInteractor.GetInteractorStyle().SetCurrentStyleToTrackballCamera()

        struct = etkStructure()
        struct.load(structurefile)
        rep = etkRepresentation()
        rep.SetStructure(struct)
        rep.AddToRenderer(renderer)
        rep.SetColorArray(variable)

        renderer.ResetCamera()


        # -----------------------------------------------------------------------------
        # Trame
        # -----------------------------------------------------------------------------

        server = get_server(client_type = "vue2")
        ctrl = server.controller

        with SinglePageLayout(server) as layout:
            layout.title.set_text("EPIC.viewer: " + structurefile)
        
            with layout.content:
                with vuetify.VContainer(
                    fluid=True,
                    classes="pa-0 fill-height",
                ):
                    view = vtk.VtkLocalView(renderWindow)

        server.start()



#
# etkStructure 
#
class etkStructure:

    def __init__(self):
        self.ID = 0

    def load(self, filepath):

        readFromString = False

        if readFromString:
            # filter out the unmapped points
            indata = ""
            print("Loading: " + structurefile)
            with open(structurefile, 'r') as infile:
                nextline = infile.readline()
                    # strip so that values are clean
                cleanline = nextline.strip()
                columns = cleanline.split(',')
                indata += nextline
                if 'unmapped' in columns:
                    colid = columns.index('unmapped')
                    nextline = infile.readline()
                    while nextline:
                            # strip so that values are clean
                        cleanline = nextline.strip()
                        values = cleanline.split(',')
                        if int(values[colid]) == 0:
                            indata += nextline 
    
                        nextline = infile.readline()
                else:
                    # there is not 'unmapped' column
                    nextline = infile.readline()
                    while nextline:
                        indata += nextline 
                        nextline = infile.readline()

        # read input csv file
        self.reader = vtkDelimitedTextReader()

        if readFromString:
            self.reader.SetInputString(indata)
            self.reader.SetReadFromInputString(True)
        else:
            self.reader.SetFileName(filepath)

        self.reader.DetectNumericColumnsOn()
        self.reader.SetFieldDelimiterCharacters(',')
        self.reader.SetHaveHeaders(True)

        # convert to poly data
        self.pd = vtkTableToPolyData()
        self.pd.SetInputConnection(self.reader.GetOutputPort())
        self.pd.SetXColumn('x')
        self.pd.SetYColumn('y')
        self.pd.SetZColumn('z')
        self.pd.Update()
            # create and add the lines to the dataset
        lines = vtkCellArray()
        for i in range(0, self.pd.GetOutput().GetPoints().GetNumberOfPoints()):
            lines.InsertNextCell(2, [i, i+1])
        self.pd.GetOutput().SetLines(lines)


#
# etkRepresentation 
#
class etkRepresentation:

    def __init__(self):
        self.ID = 0

    def SetStructure(self, structure):
        self.structure = structure

            # mapper
        self.pdMapper = vtkPolyDataMapper()
        self.pdMapper.SetInputConnection(self.structure.pd.GetOutputPort())
        self.pdMapper.GetLookupTable().SetHueRange(0.667, 0.0)
        self.pdMapper.SetScalarVisibility(True)
        self.pdMapper.SetScalarModeToUsePointFieldData()
            # actor
        self.pdActor = vtkActor()
        self.pdActor.SetMapper(self.pdMapper)

        # put a sphere at each point
            # sphere glyphs
        self.sphere = vtkSphereSource()
        self.sphere.SetPhiResolution(10)
        self.sphere.SetThetaResolution(10)
        self.sphere.SetRadius(0.05)
        self.sphere.Update()
            # connect glyph to polydata
        self.spheres = vtkGlyph3D()
        self.spheres.SetInputConnection(self.structure.pd.GetOutputPort())
        self.spheres.SetSourceConnection(self.sphere.GetOutputPort())
        self.spheres.SetColorModeToColorByScalar()
        self.spheres.Update()

            # mapper
        self.sMapper = vtkPolyDataMapper()
        self.sMapper.SetInputConnection(self.spheres.GetOutputPort())
        self.sMapper.GetLookupTable().SetHueRange(0.667, 0.0)
        self.sMapper.SetScalarVisibility(True)
        self.sMapper.SetScalarModeToUsePointFieldData()
            # actor
        self.sActor = vtkActor()
        self.sActor.SetMapper(self.sMapper)

        # cube axes
            # actor
        self.cActor = vtkCubeAxesActor()
        
        # scalar bar
        self.scalar_bar = vtkScalarBarActor()
        self.scalar_bar.SetOrientationToVertical()
        self.scalar_bar.SetLookupTable(self.sMapper.GetLookupTable())
        self.scalar_bar.SetMaximumHeightInPixels(50)

    def SetRadius(self, radius):
        self.sphere.SetRadius(radius)

    def SetColorArray(self, name):
        array = self.structure.pd.GetOutput().GetPointData().GetArray(name)
        _min, _max = array.GetRange()

        self.pdMapper.SelectColorArray(name)
        self.pdMapper.SetScalarRange(_min, _max)
        self.sMapper.SelectColorArray(name)
        self.sMapper.SetScalarRange(_min, _max)
        self.scalar_bar.SetTitle(name)

    def AddToRenderer(self, renderer):
        renderer.AddActor(self.sActor)
        renderer.AddActor(self.pdActor)
        renderer.AddActor(self.cActor)
        self.cActor.SetBounds(self.sActor.GetBounds())
        self.cActor.SetCamera(renderer.GetActiveCamera())
        self.cActor.SetFlyModeToOuterEdges()

        renderer.AddActor2D(self.scalar_bar)
