"""
Created on Mon, March 5.

Author: Manuel Schmid
Manuel.Schmid@Uni-Tuebingen.de


Class uses a rectangular grid DEM and calculates the topographic position index
(TPI) and a landform classification (after Weiss 2001). The functions to run the class
on a normal .csv-style input dem are theoretically there (see in the code marked as
deprecated) but the main usage is to load it as a landlab module.

Class has a landlab style main method "run_one_step(scalefactor)" which can be
used in the main loop to contiously write the tpi index and the landscape class
to the RasterModelGrid Instance in landlab, which is neat because then you can use
the standart NetCDF output to write data.
"""

import numpy as np
from landlab import Component
from scipy.ndimage.filters import generic_filter
import math
import logging

class landformClassifier(Component):
    """classify a DEM in different landform, according to slope, elevation and aspect"""

    def __init__(self, grid):
        """
        Constructor for the component.
        """
        self._grid = grid #loads the main dem
        self._dem = self._grid.at_node['topographic__elevation']
        self._slope = np.arctan(self._grid.at_node['topographic__steepest_slope'])
        # convert slope map to degrees for CLASSIFICATION
        self._slope = np.rad2deg(self._slope)

        self._aspect = self._grid.calc_aspect_at_node()

        #set up landlab grid structure.
        self._grid.add_zeros('node', 'topographic_position__index')
        self._grid.add_zeros('node', 'topographic_position__class')
        self._grid.add_zeros('node', 'elevation__ID')
        self._grid.add_zeros('node', 'landform__ID')
        self._grid.add_zeros('node', 'aspectSlope')
        self._grid.add_zeros('node', 'aspect')
        self._grid.add_zeros('node', 'slope_degrees')

        #Flags
        self._tpiTYPE = ''


    """
    ----------------------------------------------------------------------------
    HERE ARE FUNCTIONS WHICH ARE TUNED FOR USE IN LANDLAB
    ----------------------------------------------------------------------------
    """

    def reshapeGrid(self, nrows, ncols):

        """
        a little bit hacky right now. Landlab uses a 1D-array of elevation values
        to save data structure but we need a grid structure to run the landform
        classifier

        inputs:
            - grid : native landlab grid
            - nrows, ncols : number of rows and columns in grid
        """

        _gridRaw    = self._dem
        _slopeRaw   = self._slope
        _aspectRaw  = self._aspect

        _gridRe     = _gridRaw.reshape(nrows, ncols)
        _slopeRe    = _slopeRaw.reshape(nrows,ncols)
        _aspectRe   = _aspectRaw.reshape(nrows, ncols)
        self._dem   = _gridRe
        self._slope = _slopeRe
        self._aspect = _aspectRe

        return _gridRe, _slopeRe, _aspectRe

    def create_kernel(self,radius=2, invert=False):
        """Define a kernel
        After C.Werners lpjguesstools
        """

        if invert:
            value = 0
            k = np.ones((2*radius+1, 2*radius+1))
        else:
            value = 1
            k = np.zeros((2*radius+1, 2*radius+1))

        y,x = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x**2 + y**2 <= radius**2
        k[mask] = value

        return k

    def calcAspect(self):
        """
        Uses build-in landlab candy to get aspect values of the model grid
        """

        aspect = self._grid.calc_aspect_at_node()
        #this was moved to the .reshapeGrid() function for consitency.
        #need to reshape here directly...
        #aspect = aspect.reshape(self._grid.number_of_node_rows,
        #             self._grid.number_of_node_columns)
        self._aspect = aspect.reshape(self._grid.number_of_node_rows, self._grid.number_of_node_columns)

    def writeAspectToGrid(self):
        """
        writes aspect values to landlab-data-field
        """

        _aspectFlat = self._aspect.flatten().astype(int)
        self._grid.at_node['aspect'] = _aspectFlat
        self._grid.at_node['aspect'][self._grid.boundary_nodes] = 0

    def classifyAspect(self, classNum = 4):
        #print('aspect classifier was run!')
        """
        Classifies aspect into aspect_classes

        Can use classType = 8:
                    1.N
                    2.NE
                    3.E
                    4.SE
                    5.S
                    6.SW
                    7.W
                    8.NW

                classType = 4:
                    1.N
                    2.E
                    3.S
                    4.W
                """

        #create aspectClassArray
        _dem = self._grid
        (rows,cols) = np.shape(_dem)
        self._aspectClass = np.zeros((rows,cols))

        aspect_list = self._aspect.flatten().tolist()

        logging.debug("classifyAspect(), self._aspect: min: {}, max: {}".format(min(aspect_list), max(aspect_list)))
        # logging.debug("classifyAspect(), set(self._aspect): {}".format(set(aspect_list)))

        if classNum == '8':
            #define value breaks for aspect classes
            aspectClass = {1 : [0, 22.5],       #N
                            2 : [22.5, 67.5],    #NE
                            3 : [67.5,  112.6],  #E
                            4 : [112.5, 157.6],  #SE
                            5 : [157.5, 202.6],  #S
                            6 : [202.5, 247.6],  #SW
                            7 : [247.5, 292.6],  #W
                            8 : [292.5, 337.6],  #NW
                            9 : [337.5, 360]}    #N

            #check which node belongs to which class
            for nrow in range(rows):
                for ncol in range(cols):
                    for i,j in zip(aspectClass,aspectClass.values()):
                        if j[0] <= self._aspect[nrow,ncol] <= j[1]:
                            aspectC = i
                            if aspectC == 9:
                                aspectC = 1
                            self._aspectClass[nrow,ncol] = aspectC

        elif classNum == '4':
            #define value breaks for aspect classes
            aspectClass = {1 : [0,   45],       #N
                            2 : [45,  135],      #E
                            3 : [135, 225],      #S
                            4 : [225, 315],      #W
                            5 : [315, 360]}      #N

            #check which node belongs to which class
            for nrow in range(rows):
                for ncol in range(cols):
                    for i,j in zip(aspectClass,aspectClass.values()):
                        if j[0] <= self._aspect[nrow,ncol] <= j[1]:
                            aspectC = i
                            if aspectC == 5:
                                aspectC = 1
                            self._aspectClass[nrow,ncol] = aspectC

        else:
            raise NameError('This is not a correct argument. Check function docstring')



    def calculate_tpi(self, scalefactor, res=30, TYPE='SIMPLE'):
        """Classify DEM to tpi300 array according to Weiss 2001
        Modified after C. Werner, lpjguesstools.

        Be aware that this method is not per-se using landlab native methods
        but relies on import from landlab data and calculation by numpy.
        This means that the boundarynodes, which in landlab are marked as inactive
        and will not be considered by the landlab.imshow_grid method, are not inactive
        here.
        This mapping and conversion happens in the function ".writeTpiToGrid"
        """

        # Parameters:
        # - scalefactor: outerradius in map units (300 := 300m)
        # - res: resolution of one pixel in map units (default SRTM1: 30m)
        # - dx: cell size

        #data structures moved here for convenience
        NODATA = -9999

        # inner and outer tpi300 kernels
        k_smooth = self.create_kernel(radius=2)

        radius_outer = int(math.ceil(scalefactor / float(res)))
        radius_inner = int(math.floor((scalefactor - 150)/float(res)))

        k_outer = self.create_kernel(radius=radius_outer)
        #k_inner = self.create_kernel(radius=radius_inner, invert=True)
        if radius_inner > 0:
            k_inner = self.create_kernel(radius=radius_inner, invert=True)
        else:
            raise ValueError("Negative radius of inner kernel. Adjust Scalefactor.")

        x = y = int((k_outer.shape[0] - k_inner.shape[0]) / 2)
        k_outer[x:x+k_inner.shape[0], y:y+k_inner.shape[1]] = k_inner


        # compute tpi
        tpi = self._dem - generic_filter(self._dem, np.mean, footprint=k_outer, mode="reflect") + 0.5
        tpi = generic_filter(tpi, np.mean, footprint=k_smooth, mode="reflect").astype(int)

        logging.debug("calculate_tpi(), set(tpi): {}".format(set(tpi.flatten().tolist())))

        tpi_classes = np.ones( tpi.shape ) * NODATA

        if TYPE == 'WEISS':
            self._tpiTYPE = "WEISS"
            # values from poster, results in strange NaN values for some reason beyond me.
            mz10, mz05, pz05, pz10 = np.percentile(tpi, [100-84.13, 100-69.15, 69.15, 84.13])
            #TODO: check if this should be a decision tree (we have unclassified cells)
            tpi_classes[(tpi > pz10)]                                   = 1 # ridge
            tpi_classes[((tpi > pz05)  & (tpi <= pz10))]                = 2 # upper slope
            tpi_classes[((tpi > mz05)  & (tpi <  pz05) & (self._slope >  5))] = 3 # middle slope
            tpi_classes[((tpi >= mz05) & (tpi <= pz05) & (self._slope <= 5))] = 4 # flats slope
            tpi_classes[((tpi >= mz10) & (tpi <  mz05))]                = 5 # lower slopes
            tpi_classes[(tpi < mz10)]                                   = 6 # valleys

        # simplified:
        if TYPE == 'SIMPLE':
            self._tpiTYPE = 'SIMPLE'
            # according to Tagil & Jenness (2008) Science Alert doi:10.3923/jas.2008.910.921
            mz10, pz10 = np.percentile(tpi, [100-84.13, 84.13])
            tpi_classes[(tpi >= mz10)]                               = 1 # hilltop
            tpi_classes[(tpi >= mz10) & (tpi < pz10) & (self._slope >= 6)] = 3 # mid slope
            tpi_classes[(tpi >  mz10) & (tpi < pz10) & (self._slope  < 6)]  = 4 # flat surface
            tpi_classes[(tpi <= mz10)]                               = 6 # valley

        self._tpiClasses = tpi_classes
        logging.debug("calculate_tpi(), set(tpi_classes): {}".format(set(tpi_classes.flatten().tolist())))
        slope_list = self._slope.flatten().tolist()
        logging.debug("calculate_tpi(), min(self._slope): {}, max(self._slope): {}".format(min(slope_list), max(slope_list)))
        self._tpi = tpi

    def writeTpiToGrid(self):
        """
        Function that takes tpi-values and tpi classes and writes it back to
        Landlab Grid nodes
        """

        #flatten array to make it landlab accessible
        _tpiFlat = self._tpi.flatten().astype(int)
        _tpiClassFlat = self._tpiClasses.flatten().astype(int)

        #write data to array
        for struct,data in zip(['topographic_position__index', 'topographic_position__class'],
                        [_tpiFlat, _tpiClassFlat]):
            self._grid.at_node[struct] = data
            self._grid.at_node[struct][self._grid.boundary_nodes] = 0

    def updateGrid(self):
        """
        Call this to reload Grid-parameters like slope and elevation.
        This takes time. Is there a more efficient way to do this?
        Is this even needed? Is this the real life or is this just fantasy.
        caught in a bugslide, no escape from morphology.
        """

        self._dem = self._grid.at_node['topographic__elevation']
        self._slope = np.arctan(self._grid.at_node['topographic__steepest_slope'])
        # convert slope map to degrees for CLASSIFICATION
        self._slope = np.rad2deg(self._slope)
        self._grid.at_node['slope_degrees'] = self._slope
        self._aspect = self._grid.calc_aspect_at_node()

    def createElevationID(self, dem, minimum, maximum, step):
        elevationID = np.zeros(np.shape(dem)) #creates ID array
        elevationSteps = np.arange(minimum, maximum, step) #creates elevation-step array
        logging.debug("createElevationID(), min: {}, max: {}, step: {}".format(minimum, maximum, step))
        counterID = 1 #starts at 1 for lowester elevation class

        for i in elevationSteps:
            index = np.where((dem >= i) & (dem < i+step))
            elevationID[index] = counterID
            logging.debug("createElevationID(), counterID: {}".format(counterID))
            if index[0].size > 0 :
                logging.debug("createElevationID(), counterID: i: {}".format(i))
            counterID += 1

        self._elevationID = elevationID
        self._grid.at_node['elevation__ID'] = elevationID

        return elevationID

    def createLandformID(self):
        """
        creates a per-node grid with a 3-integer landform-ID.

        ID-system is:
            [elevationID, slopeID, aspectID]

            elevationID:
                1 = [minium:step[i]] e.g    1 = [0:200]
                                            2 = [200:400]
                                            3 = [400 : 600]

            slopeID: TPI Slope ID, see TPI function

            aspectID:
            right now it only works with the 4-class aspect ID system for consistency with LPJGuess
                1 N
                2 E
                3 S
                4 W

        """

        #this is not very memory efficient and was just done for ease of coding
        #can be changed later but due to pythons fancy indexing I guess this is still
        #ok.
        _slopeID     = self._tpiClasses.flatten().astype(int)
        _aspectID    = self._aspectClass.flatten().astype(int)
        _elevationID = self._elevationID.flatten().astype(int)

        logging.debug("landformClassifier, createLandformID, set(_slopeID): {}".format(set(_slopeID.tolist())))
        logging.debug("landformClassifier, createLandformID, set(_aspectID): {}".format(set(_aspectID.tolist())))
        logging.debug("landformClassifier, createLandformID, set(_elevationID): {}".format(set(_elevationID.tolist())))

        #check which tpi_type was used and adjust the matrix with aspect-dependend landform

        if self._tpiTYPE == 'WEISS':
            lfClasses = [2,3,5]
        elif self._tpiTYPE == 'SIMPLE':
            lfClasses = [3]

        logging.debug("lfClasses: {}".format(lfClasses))

        lfIndex_set = set()

        for i in range(len(self._grid.at_node['landform__ID'])):
            if _slopeID[i] in lfClasses:
                lfIndex = "{}{}{}".format(_elevationID[i], _slopeID[i], _aspectID[i])
            else:
                lfIndex = "{}{}0".format(_elevationID[i], _slopeID[i])

            self._grid.at_node['landform__ID'][i] = int(lfIndex)
            lfIndex_set.add(lfIndex)

        logging.debug("landformClassifier, createLandformID, lfIndex: {}".format(lfIndex_set))

    def calc_asp_slope(self):
        """
        calculate some strange aspect-slope-metric that lpjguess needs and
        nobody understand excepts some random climate-dude from senckenberg
        """
        _aspSlope = self._slope * np.abs(np.cos(np.radians(self._aspect)))

        return _aspSlope

    def write_asp_slope_to_grid(self):
        """
        Calls the calc_asp_slope function and writes the returned values
        back to the landlab grid
        """

        _aspSlope = self.calc_asp_slope()
        _aspSlopeFlat = _aspSlope.flatten().astype(int)

        self._grid.at_node['aspectSlope'] = _aspSlopeFlat
        self._grid.at_node['aspectSlope'][self._grid.boundary_nodes] = 0

    def run_one_step(self, elevationStepBin, scalefact, classtype, max_elevation):
        """
        Landlab style wrapper function which is to be called in the main-model-loop

        inputs:
            elevationBin : bin-size for elevation Id
            scalefact: scalefactor for classification donut
            classtype: 'SIMPLE' or 'WEISS', after Weiss, 2001
            max_elevation: possible maximum elevation
        """
        self.updateGrid()
        self.reshapeGrid(nrows = self._grid.number_of_node_rows,
                        ncols = self._grid.number_of_node_columns)
        self.calculate_tpi(scalefact, res = self._grid.dx, TYPE = classtype)
        self.calcAspect()
        self.write_asp_slope_to_grid()
        self.writeAspectToGrid()
        self.classifyAspect(classNum = '4')
        self.createElevationID(self._dem, 0, max_elevation, elevationStepBin)
        self.createLandformID()
        self.writeTpiToGrid()
