"""Simulate detachment limited sediment transport.

Landlab component that simulates detachment limited sediment transport is more
general than the stream power component. Doesn't require the upstream node
order, links to flow receiver and flow receiver fields. Instead, takes in
the discharge values on NODES calculated by the OverlandFlow class and
erodes the landscape in response to the output discharge.

As of right now, this component relies on the OverlandFlow component
for stability. There are no stability criteria implemented in this class.
To ensure model stability, use StreamPowerEroder or FastscapeEroder
components instead.

.. codeauthor:: Jordan Adams

Examples
--------
>>> import numpy as np
>>> from landlab import RasterModelGrid
>>> from landlab.components import DetachmentLtdErosion

Create a grid on which to calculate detachment ltd sediment transport.

>>> grid = RasterModelGrid((4, 5))

The grid will need some data to provide the detachment limited sediment
transport component. To check the names of the fields that provide input to
the detachment ltd transport component, use the *input_var_names* class
property.

Create fields of data for each of these input variables.

>>> grid.at_node['topographic__elevation'] = np.array([
...     0., 0., 0., 0., 0.,
...     1., 1., 1., 1., 1.,
...     2., 2., 2., 2., 2.,
...     3., 3., 3., 3., 3.])

Using the set topography, now we will calculate slopes on all nodes.


>>> grid.at_node['topographic__slope'] = np.array([
...     -0.        , -0.        , -0.        , -0.        , -0,
...      0.70710678,  1.        ,  1.        ,  1.        ,  0.70710678,
...      0.70710678,  1.        ,  1.        ,  1.        ,  0.70710678,
...      0.70710678,  1.        ,  1.        ,  1.        ,  0.70710678])


Now we will arbitrarily add water discharge to each node for simplicity.
>>> grid.at_node['surface_water__discharge'] = np.array([
...     30., 30., 30., 30., 30.,
...     20., 20., 20., 20., 20.,
...     10., 10., 10., 10., 10.,
...      5., 5., 5., 5., 5.])

Instantiate the `DetachmentLtdErosion` component to work on this grid, and
run it. In this simple case, we need to pass it a time step ('dt')

>>> dt = 10.0
>>> dle = DetachmentLtdErosion(grid)
>>> dle.erode(dt=dt)

After calculating the erosion rate, the elevation field is updated in the
grid. Use the *output_var_names* property to see the names of the fields that
have been changed.

>>> dle.output_var_names
('topographic__elevation',)

The `topographic__elevation` field is defined at nodes.

>>> dle.var_loc('topographic__elevation')
'node'


Now we test to see how the topography changed as a function of the erosion
rate.

>>> grid.at_node['topographic__elevation'] # doctest: +NORMALIZE_WHITESPACE
array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.99993675,  0.99991056,  0.99991056,  0.99991056,  0.99993675,
        1.99995528,  1.99993675,  1.99993675,  1.99993675,  1.99995528,
        2.99996838,  2.99995528,  2.99995528,  2.99995528,  2.99996838])

"""

from landlab import Component
from landlab import ModelParameterDictionary, CLOSED_BOUNDARY, Component
import pylab
import numpy as np
from matplotlib import pyplot as plt
from landlab.field.scalar_data_fields import FieldError
UNDEFINED_INDEX = np.iinfo(np.int32).max


class vegetationLimitErosion(Component):

    """Landlab component that simulates detachment-limited river erosion.

    This component calculates changes in elevation in response to vertical
    incision.
    """

    _name = 'DetachmentLtdErosion'

    _input_var_names = (
        'topographic__elevation',
        'topographic__slope',
        'surface_water__discharge',
    )

    _output_var_names = (
        'topographic__elevation',
    )

    _var_units = {
        'topographic__elevation': 'm',
        'topographic__slope': '-',
        'surface_water__discharge': 'm^3/s',
    }

    _var_mapping = {
        'topographic__elevation': 'node',
        'topographic__slope': 'node',
        'surface_water__discharge': 'node',
    }

    _var_doc = {
        'topographic__elevation': 'Land surface topographic elevation',
        'topographic__slope': 'Slope of ',
        'surface_water__discharge': 'node',
    }

    def __init__(self, grid, K_sp = 0.00002, m_sp = 0.5, n_sp = 1,
                 uplift_rate = 0.0, entraiment_threshold = 0.0,
                 vegetation__density = None, **kwds):
        """Calculate detachment limited erosion rate on nodes.

        Landlab component that generalizes the detachment limited erosion
        equation, primarily to be coupled to the the Landlab OverlandFlow
        component.

        This component adjusts topographic elevation and is contained in the
        landlab.components.detachment_ltd_sed_trp folder.

        Parameters
        ----------
        grid : RasterModelGrid
            A landlab grid.
        K_sp : float, optional
            K in the stream power equation (units vary with other parameters -
            if used with the de Almeida equation it is paramount to make sure
            the time component is set to *seconds*, not *years*!)
        m_sp : float, optional
            Stream power exponent, power on discharge
        n_sp : float, optional
            Stream power exponent, power on slope
        uplift_rate : float, optional
            changes in topographic elevation due to tectonic uplift
        entrainment_threshold : float, optional
            threshold for sediment movement
        """
        super(vegetationLimitErosion, self).__init__(grid, **kwds)

        self.stream_power_erosion = self.grid.zeros('node', dtype = float)
        self.grid.at_node['stream_power_erosion'] = self.grid.zeros('node',dtype = float)
        self.K = K_sp
        self.m = m_sp
        self.n = n_sp
        #self.mnb = 99
        #self.mnv = 22


        #self.I = self._grid.zeros(at='node')
        self.I = self.grid.zeros(at='node')
        self.uplift_rate = uplift_rate
        self.entraiment_threshold = entraiment_threshold

        self.dzdt = self.grid.zeros(at='node')

        if vegetation__density is not None:
            if type(vegetation__density) is not str:
                self.vd = vegetation__density
            else:
                self.vd = self.grid.at_node[vegetation__density]
        else:
                raise KeyError("vegetation__density must be provided to the " +
                                                   "component you asshole")



    def erode(self, dt, drainage_area='drainage_area',
              slope='topographic__steepest_slope',
              flow_receiver='flow__receiver_node',
              node_order_upstream='flow__upstream_node_order',
              link_node_mapping='flow__link_to_receiver_node'):
        """Erode into grid topography.

        For one time step, this erodes into the grid topography using
        the water discharge and topographic slope.

        The grid field 'topographic__elevation' is altered each time step.

        Parameters
        ----------
        dt : float
            Time step.
        discharge_cms : str, optional
            Name of the field that represents discharge on the nodes, if
            from the de Almeida solution have units of cubic meters per second.
        slope : str, optional
            Name of the field that represent topographic slope on each node.
        """

        #CREATES NODES ORDERING ACCORDING TO STREAM_POWER-CLASS
        upstream_order_IDs = self.grid['node']['flow__upstream_node_order']
        #node_order_upstream = self.grid.at_node['node_order_upstream']
        defined_flow_receivers = np.not_equal(self._grid['node']['flow__link_to_receiver_node'], UNDEFINED_INDEX)
        flow_link_lengths = self.grid._length_of_link_with_diagonals[self._grid['node']['flow__link_to_receiver_node'][defined_flow_receivers]]
        active_nodes = np.where(self.grid.status_at_node != CLOSED_BOUNDARY)[0]
        flow_receivers = self.grid['node']['flow__receiver_node']

        try:
            S = self.grid.at_node[slope]
        except FieldError:
            raise ValueError('missing field for slope')

        if type(drainage_area) is str:
            node_A = self.grid.at_node[drainage_area]
        else:
            node_A = drainage_area


        A_to_m = np.power(node_A[active_nodes], self.m)
        #S_to_n = np.power(S, self.n)
        S[S == 0.0] += 0.001
        S_to_n = S[active_nodes]**self.n

        #needed:
        self.AqDens = 1000.0 #Density of Water [Kg/m^3]
        self.grav   = 9.81   #Gravitational Acceleration [N/Kg]
        self.n_soil = 0.025  #Mannings roughness for soil [-]
        self.n_VRef = 0.7    #Mannings Reference Roughness for Vegi [-]
        self.v_ref  = 0.9    #Reference Vegetation Density
        w      = 1.    #Some scaling factor for vegetation [-?]


        #something new
        ##UPPER PART OF "NEW-K-TERM"
        nSoil_to_15 = np.power(self.n_soil, 1.5)
        self.Ford    = self.AqDens * self.grav * nSoil_to_15 ##STAYS CONSTANT!
        ##LOWER PART OF "LOW-K-TERM"
        n_v_frac = self.n_soil + (self.n_VRef*(self.vd[active_nodes]/self.v_ref)) #self.vd = VARIABLE!
        n_v_frac_to_w = np.power(n_v_frac, w)
        self.Prefect = np.power(n_v_frac_to_w, 0.9)

        self.Kv = self.K * (self.Ford/self.Prefect)

        #Calculate the stream-power at the active_nodes:
        self.stream_power_active_nodes = self.Kv * dt * A_to_m * S_to_n




        #something old
        #self.I = (self.Kv * (A_to_m * S_to_n - self.entraiment_threshold))
        self.stream_power_erosion[active_nodes] = self.stream_power_active_nodes
        self.grid.at_node['stream_power_erosion'][active_nodes] = self.stream_power_erosion
        I = (self.grid.at_node['stream_power_erosion'] - self.entraiment_threshold).clip(0.)
        #self.I[self.I < 0.0] = 0.0

        #INTRODUCE NODE ORDERING
        node_z = self.grid.at_node['topographic__elevation']
        elev_dstr = node_z[defined_flow_receivers]

        for i in upstream_order_IDs:
            elev_this_node_before = node_z[i]
            elev_this_node_after = (elev_this_node_before - I[i])
            elev_dstr_node_after = (elev_dstr[i] - I[defined_flow_receivers[i]])
            if elev_this_node_after < elev_dstr_node_after:
                I[i] = (elev_this_node_before - elev_dstr_node_after) * 0.99999

        self.dzdt -= I.clip(0.)
        #self.dzdt = (self.uplift_rate - self.I)

        self.grid['node']['topographic__elevation'] += self.dzdt
