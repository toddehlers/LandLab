from collections import OrderedDict
import numpy as np
import pandas as pd
import xarray as xr
import string
import sys
import math
import copy
import glob
import os

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt


import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

import xarray as xr

import matplotlib as mpl

import matplotlib.cm as cm
from matplotlib.collections import PatchCollection
from matplotlib.legend_handler import HandlerBase





# Depending on what you want to do this might suffice… It requires model output processed with my ldndc2nc post processing routines.

# It seems I (again) archived most o the stuff but some of the more ad-hoc scripts slipped through the cracks. 

# The script is called like this:
# python plot_timeseries.py lpjguess_out.nc lpjguess_biome_out.nc landlab_output.nc lfid

# The arguments are:
# 1) lpjguess_out.nc: this is the “regular” netcdf output file produced by ldndc2nc 
# 2) lpjguess_biome_out.nc: this is the “biomization” output file (also produced by ldndc2nc I think)
# 3) the landlab output file (includes erosion for the plot I think)
# 4) the site id (integer) that should be plotted: 0-3 for 4 ES sites
# 5) landformid

# If 5 is given, the plot is done for the dominant landform id in the cell, otherwise for an area average.
# You don’t want to have the average since this smooths out the PFT transitions so you can’t see much I think...

# One note: 
# Unfortunately, library versions are not pinned in the script (it’s just a plain script and not a proper package) so you might
# encounter some errors/ API changes if you install current python dependencies (especially xarray had some bigger changes).
# The script was created end of 2018 so xarray version at the time was well below 0.13 where some breaking changes happened.
# Try if the code runs as is, but maybe you need to install a xarray version prior to release 0.13 







def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)

def majority(x, axis=0):
    return np.argmax(np.bincount( x.astype(np.int)))

# modify linewidth of hatches
matplotlib.rcParams['hatch.linewidth'] = 0.5  # previous pdf hatch linewidth
matplotlib.rcParams['axes.labelsize'] = 10
matplotlib.rcParams['font.size']= 10

# modify legend patch size
#matplotlib.rcParams['legend.handlelength'] = 2
#matplotlib.rcParams['legend.handleheight'] = 1 
#matplotlib.rcParams['legend.numpoints'] = 1
# this should be imported


# NEW CHILE BIOMIZATION


D_cols2  = OrderedDict([('TeBE', "#006400"),
                ('TeBEX', "#8f8f00"),
                ('TeBS',  "#90ee90"),
                ('TeSh', "#cdcd00"),
                ('TeNE', "#6b8e23"),
                ('BSh', "#ffaeb9"),
                ('BBS',  "#314d5a"), 
		('BBE', "#45a081"),
                ('Grass', "#d2b48c")])


INCLUDE_GRASS = True

# map individual pfts to pft groups with symbol
D_pfts = OrderedDict()
D_pfts['TeBE']  = ('TeBE_itm', 'TeBE_tm')
D_pfts['TeBS']  = ('TeBS_itm', 'TeBS_tm')
D_pfts['TeBEX'] = ('TeBE_itscl',)
D_pfts['TeSh']  = ('TeE_s','TeR_s')
D_pfts['TeNE']  = ('TeNE',)
D_pfts['BSh'] = ('BE_s',) #  'BS_s')
D_pfts['BBS'] = ('BBS_itm',)
D_pfts['BBE'] = ('BBE_itm',)

if INCLUDE_GRASS:
    D_pfts['Grass'] = ('C3G',)


def shift_legend(leg, ax, x_offset, y_offset):
    """Shift the legend from the current position
    """
    # Get the bounding box of the original legend
    bb = leg.get_bbox_to_anchor().inverse_transformed(ax.transAxes)

    # Change to location of the legend. 
    bb.x0 += x_offset
    bb.x1 += x_offset
    bb.y0 -= y_offset
    bb.y1 -= y_offset
    leg.set_bbox_to_anchor(bb, transform = ax.transAxes)



class AnyObject(object):
    """Store color and patch attributes for legend
    """
    def __init__(self, color, left, right=None):
        # valid values:
        self.color = color
        self.left  = left
        self.right = right

class AnyObjectHandler(HandlerBase):
    """Custom legend handler for PFT plot
    """
    def create_artists(self, legend, ohandle,
                    x0, y0, width, height, fontsize, trans):
    
        r = []

        # if only one patch, center it
        if not ohandle.right:
            #lpos = (x0 + width*0.275, y0)
            the_width = width
            rpos = (x0, y0)
        else:
            the_width = width * 0.475
            rpos = (x0+0.525*width,y0)

        lpos = (x0, y0)
            
        r.append( mpatches.Rectangle(lpos, the_width, height, ec='black',
                   linestyle='-', fc=ohandle.color, lw=0.1, hatch=ohandle.left) )
        
        if ohandle.right:
            r.append( mpatches.Rectangle(rpos, the_width, height, ec='black',
                   linestyle='-', fc=ohandle.color, lw=0.1, hatch=ohandle.right ) )

        return r



# PLOT TYPE
bg_color = 'white'

if len(sys.argv) > 1:
    bg_color = sys.argv[1]

if bg_color == 'black':
    fg_color = 'white'
else:
    fg_color = 'black'
    



# read sites location
#sites      = pd.read_csv('ES_SiteLocations_checked.csv', sep=';')
siteName   = ['Nahuelbuta', 'La Campana', 'St. Gracia', 'Pan de Azucar']
siteNameShort = ['N','LC','SG','PdA']

VARS       = ['temp', 'PRECT', 'FSDS']

Dvar = {'temp': "$\mathregular{T_{avg}\ [\degree C]}$",
        'prec': "$\mathregular{Precip.\ [mm]}$",
        'runoff': "$\mathregular{Runoff\ [mm]}$",
        'fpc': "FPC [%]",
        'vcf': "Fractional Cover [%] (VCF)",
        'co2': 'CO$_2$ [ppm]',
        'fire': "Fire RI [yr]",
        'erosion': '$\mathregular{Erosion\ [mm\ y^{-1}]}$'}

def clean(s):
    _s = s.replace('.','').replace(' ', '_')
    return _s

def smoothTriangle(data, degree, dropVals=False):
    """performs moving triangle smoothing with a variable degree."""
    """note that if dropVals is False, output length will be identical
    to input length, but with copies of data at the flanking regions"""
    triangle=np.array(range(degree)+[degree]+range(degree)[::-1])+1
    smoothed=[]
    for i in range(degree,len(data)-degree*2):
        point=data[i:i+len(triangle)]*triangle
        smoothed.append(sum(point)/sum(triangle))
    if dropVals: return smoothed
    smoothed=[smoothed[0]]*(degree+degree/2)+smoothed
    while len(smoothed)<len(data):smoothed.append(smoothed[-1])
    return smoothed


def calc_delta(times):
    """ create time-slices for: 22000, 18000, 14000, 10000, 6000, 2000, 1990 BP """
    btime = 22000   # BP
    if type(times) == list:
        idx = [(t+btime)*12 for t in times]
    else:
        idx = times+btime*12
    return idx


def custom_stackplot(axis, x, ys, colors, hatches, **kwargs):
    """Custom stackplot (with individual hatches)"""    
    prev_d = None
    cd1 = np.array([0] * len(ys[0]))
    for i, d2 in enumerate(ys):
        cd2 = cd1 + np.array(d2)
        axis.fill_between(x, cd1, cd2, facecolor=colors[i], edgecolor='black', hatch = hatches[i], 
				**kwargs)  ## < removes facecolor
        cd1 = cd2


def main(fdataname, fbiomename, flandlab, p, maxlfid=False):

    ADD_BIOME = True
    INCLUDE_GRASS = True
    DO_MAXLFID = maxlfid
    DROP_LABELS = False

    ds2  = xr.open_dataset(fdataname, decode_times=False) #.sel(time=slice(-20010,1989))
    ds2['time'] = ds2['time'] - 1950 # convert to BP values

    if maxlfid:
        D_mapEle={0: np.array([6]), 1: np.array([2,3]), 2: np.array([3]), 3: np.array([2])}
        print 'Using dominant landform only'
        print 'Searching in elevation bracket'
        # select largest landform in this cell
        # get dominant land_id
        bracket = D_mapEle[p]*100
        lf_sel = (ds2['lf_id'] >= min(bracket)) & (ds2['lf_id'] < max(bracket)+100)
        fid=np.argmax(ds2['fraction'].where( lf_sel ))
        the_lf_id = ds2.coords['lf_id'].values[fid]
        the_lf_fr = float(np.nanmax(ds2['fraction'].where( lf_sel ).values))
        ds2  = ds2.sel(lf_id=the_lf_id, drop=True)

        print p, the_lf_id, the_lf_fr

    if ADD_BIOME:
        from lpjguesstools.lgt_biomize.biomes_earthshape import biomes, biome_color
        dsb2 = xr.open_dataset(fbiomename, decode_times=False)['biome'].to_dataset(name='biome')
        dsb2['time'] = dsb2['time'] - 1950 # convert to BP values
        
        if maxlfid:
            dsb2 = xr.open_dataset(fbiomename, decode_times=False)['biome_lf'].to_dataset(name='biome')
            dsb2['time'] = dsb2['time'] - 1950 # convert to BP values
            dsb2 = dsb2.sel(lf_id=the_lf_id, drop=True)


    # landlab get erosion rate
    ds_ll = xr.open_dataset(flandlab)
    lfids = np.unique(ds_ll['landform__ID'].astype('int').values)

    if the_lf_id not in lfids:
        print('LFID PROBLEM. Abort...')
    else:

        mask = ds_ll['landform__ID'].values==the_lf_id
        #v = ds['vegetation__density'].where(mask).mean(dim=['ni','nj'])
        e = (ds_ll['erosionRate'].where(mask)*1000).mean(['ni','nj'])
        #r = (ds['rainvalue'].where(mask)*10).mean()
        #Dveg[lfid].append( v.values )
        #Dero[lfid].append( e.values )
        #Drai[lfid].append( r.values )
        #Dc[lfid].append( np.count_nonzero(mask))
        print(e)

    df_erosion = pd.DataFrame({'time': np.arange(len(e)*100), 'erosion': np.repeat( e.values, 100 )})


    # .sel(time=slice(-20010,1989))['biome'].to_dataset(name='biome')

    df_co2 = pd.read_table('co2_TraCE_egu2018_35ka_const180ppm.txt', delim_whitespace=True, header=None)
    df_co2.columns = ['time', 'CO2']

    df_co2['time'] = df_co2['time'] - 1950 # convert to BP values

    doit = False
    if doit:

        # create center-interval timesteps
        start_ctime = ds.coords['time'].values[0] + (interval/2)
        end_ctime = ds.coords['time'].values[-1] - (interval/2) +1
        times_cint = np.linspace(start_ctime, end_ctime, num=len(ds['time'])/interval).astype(np.int)

        # add aggregation axis
        ds['time_c'] = xr.full_like(ds['time'], 0)
        ds['time_c'][:] = np.repeat(times_cint, interval)

        dsb['time_c'] = xr.full_like(dsb['time'], 0)
        dsb['time_c'][:] = np.repeat(times_cint, interval)

        # aggregate segments, rename time axis back to 'time'
        ds2 = ds.groupby('time_c').mean(dim='time')
        ds2.rename(dict(time_c='time'), inplace=True)

        dsb = dsb.swap_dims(dict(time='time_c'))
        del dsb.coords['time']

    print '---------'
    pfts = ds2.coords['pft'].values

    # climate vars ---
    da_temp = ds2['sp_mtemp'].mean(dim='month')
    da_prec = ds2['sp_mprec'].sum(dim='month')
    da_rad  = ds2['sp_mrad'].mean(dim='month')
    # read co2 data from file

    # pft lai vars ---
    da_lai = ds2['sp_lai']

    # landlab vars ---
    da_fpc = ds2['tot_sp_fpc']
    da_vcf_tree = ds2['VCF_Tree']
    da_vcf_nontree = ds2['VCF_NonTreeVeg']
    da_vcf_bare = ds2['VCF_Bare']

    da_fire = ds2['FireRT']
    #da_run = ds2['sp_mrunoff'].sum(dim='month')
    da_run = ds2['Runoff_Surf']

    # biomes ---
    
    if ADD_BIOME:
        da_biome = dsb2['biome']

    # build dataframe
    cols = []
    cols.append( da_temp.to_pandas().rename('tempavg') )
    cols.append( da_prec.to_pandas().rename('precsum') )
    cols.append( da_rad.to_pandas().rename('radavg') )
    #   cols.append( da_co2.to_pandas().rename('co2') )

    # add the pft lais
    for p_ in pfts:
        cols.append( da_lai.sel(pft=p_).to_pandas().rename(p_) )

    if ADD_BIOME:
        # add biome
        cols.append( da_biome.to_pandas().rename('biome') )

    # add landlab vars
    cols.append( da_fpc.to_pandas().rename('fpcavg') )
    cols.append( da_run.to_pandas().rename('runoffavg')  )

    cols.append( da_vcf_tree.to_pandas().rename('tree')  )
    cols.append( da_vcf_nontree.to_pandas().rename('nontree')  )

    cols.append( da_fire.to_pandas().rename('FireRT')  )

    syear = da_temp.coords['time'].values.min()
    eyear = da_temp.coords['time'].values.max()
    
    df_co2_b = df_co2[(df_co2.time >= syear) & (df_co2.time <= eyear)]
    df_co2_b.set_index(df_co2_b.time, inplace=True)
    cols.append( df_co2_b['CO2'])

    df_erosion_b = df_erosion[(df_erosion.time >= syear) & (df_erosion.time <= eyear)]
    df_erosion_b.set_index(df_erosion_b.time, inplace=True)
    cols.append( df_erosion_b['erosion'] )


    # create dataframe
    df = pd.concat(cols, axis=1) #.reset_index()

    if ADD_BIOME:
        df_biome = df[['biome']]

    # original dataframe done
    
    # simplify (smooth, step-wise)
    
    if len(df) > 2000:
        interval = 100
    elif len(df) > 200:
        interval = 10
    else:
        interval = None

    # METHOD = 'smooth'
    METHOD='smooth'

    if METHOD == 'smooth':
        if interval is not None:
            df = df.rolling(min_periods=1, center=True, window=interval).mean()
            if ADD_BIOME:
                df['biome'] = df_biome.rolling(min_periods=1, center=True, window=interval).apply(majority)
        else:
            if ADD_BIOME:
                df['biome'] = df_biome
    elif METHOD == 'step':
        if interval is not None:
            df = df.groupby(df.index//interval).transform('mean')
            if ADD_BIOME:
                df['biome'] = df_biome.groupby(df_biome.index//interval).transform(majority)
        else:
            if ADD_BIOME:
                df['biome'] = df_biome

    # add co2 (non-smoothed)
    df.update( df_co2_b )        
    df.reset_index(inplace=True)

    # add erosion (non-smoothed)
    df.update( df_erosion_b )
    df.reset_index(inplace=True)

    # patch x-axis coords for biome tile
    df['syear'] = df['time'] - 1 # - interval/2
    df['eyear'] = df['time'] + 1 #+ interval/2 -1
    df['Total'] = df.loc[:, pfts].sum(axis=1)

    # fix columns / add original co2
    df['FireRT'] = (1/ df.FireRT)
    bg_color = 'white'
    # ACTUAL PLOT STARTS HERE
    if bg_color == 'white':
        fig_facecolor = 'none'
    else:
        fig_facecolor = bg_color

    # dictionary for right axis
    no_of_panels = 5
    
    #df['time'] = df['time'] + 1950
    df = df[df.time < 20000]
    
    #
    # BUILD PLOT --------------------------------------
    #
    
    # create timeseries plot
    fig, ax = plt.subplots(no_of_panels, figsize=(6, 7.5), sharex=True, 
            facecolor=fig_facecolor, edgecolor=fg_color,
            gridspec_kw={'height_ratios':[0.75, 0.375, 1.125, 1.0, 0.75],
                         'hspace': 0.125})

    ax_right = {} # dictionary with right axis (key ID: value ax)

    for ID in range(no_of_panels):
        ax[ID].xaxis.set_zorder(100) 
        ax[ID].yaxis.set_zorder(100) 

    # panel co2 ---------------------------------------------
    #ID = 0
    #ax[ID].plot(df.time, df.CO2, linestyle='-', lw=1, color='black')
    #ax[ID].set_ylabel(Dvar['co2'], color='black')
    #ax[ID].tick_params('y', colors=fg_color)
    #ax[ID].set_ylim(ymin=180, ymax=300)
    #ax[ID].set_yticks([200,250, 300])

    #ax[ID].set_ylim(ymin=180, ymax=300)
    #ax[ID].set_yticks([200,250, 300])
    
    # panel 1: climate

    # panel climvars ------------------------------------------
    ID = 0
    ax[ID].plot(df.time, df.tempavg, linestyle='-', lw=1, color='red')
    ax[ID].set_ylabel(Dvar['temp'], color='black')
    ax[ID].tick_params('y', colors=fg_color)
    ax[ID].set_yticks([5, 10, 15, 20])
    ax[ID].set_ylim(ymin=5, ymax=20)

    ax_right[ID] = ax[ID].twinx()
    ax_right[ID].plot(df.time, df.precsum, linestyle='-', lw=1, color='blue')
    ax_right[ID].set_ylabel(Dvar['prec'], color='black') #, rotation=270, va='bottom')
    ax_right[ID].tick_params('y', colors=fg_color)
    ax_right[ID].set_yticks([0, 500, 1000, 1500, 2000])
    ax_right[ID].set_ylim(ymin=0, ymax=2000)

    lines = [mlines.Line2D([], [], color='red',markersize=12),
             mlines.Line2D([], [], color='blue',markersize=12)]
    labels = ['T$_{avg}$', 'Precip']


    leg = ax_right[ID].legend(lines, labels, loc='upper left', ncol=2,
                                 prop={'size':8}, frameon=True, facecolor='white',
                                 columnspacing=.6)
    leg.get_frame().set_linewidth(0.0)
    #leg.get_frame().set_facecolor('white')
    #leg.get_frame().set_alpha(1.0)
    #leg.get_frame().set_zorder(10001)
    

    # plot fire  ------------------------------------------
    ID += 1
    ax[ID].plot(df.time, df.FireRT, linestyle='-', label='Fire Return Interval [yrs]', lw=.5, color='black')
    ax[ID].set_ylabel(Dvar['fire'], color='black') #, rotation=270, va='bottom')
    ax[ID].tick_params('y', colors=fg_color)
    ax[ID].set_ylim(ymin=0, ymax=140)
    ax[ID].set_yticks([0,50,100])
    ax[ID].invert_yaxis()


    # plot smooth pfts (lai) ------------------------------------------
    ID += 1

    # all pft names
    #pft_names = da_lai.coords['pft'].values.tolist() # df.loc[0, 'TeBE_tm':'C3G'].index.values
    # correct order:
    
    pft_names = ['TeBE_itm', 'TeBE_tm', 'TeBE_itscl', 'TeBS_itm', 'TeBS_tm', 'TeE_s', 'TeR_s', 'TeNE', 'BBS_itm', 'BBE_itm', 'BE_s'] #, 'BS_s'] # BBE_itm

    if INCLUDE_GRASS:
        pft_names += ['C3G'] 


    # all pft names present at site
    pft_names_site = []
    for p_ in pft_names:
        # threshold
        if df[p_].max() >= 0.05:
            pft_names_site.append(p_)

    cols = [df[p_] for p_ in pft_names_site]


    def hatch(x):
        """Return hacth pattern for key (it, s, r, otherwise None) 
        """
        D_hs = dict(t='////', s='....', r='xxxxx')
        hatch_pattern = None
        if x in D_hs.keys():
            hatch_pattern=D_hs[x]
        return hatch_pattern

    def get_hatch(x):
        y = None
        if '_t' in x:
            y = hatch('t')
        #elif 'E_s' in x:
        #    y = hatch('e')
        elif 'R_s' in x:
            y = hatch('r')
        elif 'S_s' in x:
            y = hatch('s')
        else:
            y = None
        return y

    def invert(d):
        """Flip a dictionary around"""
        return OrderedDict( (v,k) for k in d for v in d[k] )


    def get_color(x):
        d = invert(D_pfts)
        z = d[x]
        c = D_cols2[ z ]
        return c

    _colors = [get_color(p_) for p_ in pft_names_site]
    _hatches = [get_hatch(h_) for h_ in pft_names_site]


    ax[ID].fill_between(df.time, df.Total, color='lightgray', lw=0.5)    # background fill (other)
    custom_stackplot(ax[ID], df.time, cols, _colors, _hatches, lw=0.25)
    ax[ID].plot(df.time, df.Total, linestyle='-', lw=0.5, color='black')

    ax[ID].set_ylabel("PFT [LAI]", color=fg_color)
    
    # set some sensible ylims
    #if df.Total.max() < 2:
    #    ax[ID].set_ylim(ymin=0, ymax=2)
    #else:
    ax[ID].set_ylim(ymin=0, ymax=4)


    # --- FPC cover
    # last panel FPC, RUNOFF
    ID+=1 

    # runoff
    if bg_color == 'black':
        rcolor = 'lightblue'
    else:
        rcolor = 'darkblue'

    # calculate fpc based on beer-law
    import math
    df['fpc_th'] = df['Total'].apply(lambda x: (1-math.exp(-0.5*x))*100)

    # scale partitioned fpc with fpc_th
    #

    df['tree_nontree'] = df.tree + df.nontree
    df['tree'] = (df.tree / df.tree_nontree) * df.fpc_th
    df['nontree'] = (df.nontree / df.tree_nontree) * df.fpc_th

    ax[ID].stackplot(df.time, df.tree, df.nontree, linestyle='-', edgecolor='black', 
                     lw=0.25, colors=['darkgray', 'lightgray'])
    ax[ID].set_ylabel(Dvar['fpc'], color='black')
    ax[ID].tick_params('y', colors=fg_color)
    ax[ID].set_ylim(ymin=0, ymax=100)
    ax[ID].set_yticks([0,25,50,75,100])


    ax_right[ID] = ax[ID].twinx()
    ax_right[ID].plot(df.time, df.runoffavg, linestyle='-', lw=1, color='black')
    ax_right[ID].set_ylabel(Dvar['runoff'], color='black') #, rotation=90, va='bottom')
    ax_right[ID].tick_params('y', colors=fg_color)
    ax_right[ID].set_ylim(ymin=0, ymax=549.)
    
    # set some sensible ylims
    #if (df.tree*100 + df.nontree*100).max() < 50:
    #    ax[ID].set_ylim(ymax=54.99)
    #    bmax = 4.99
    #    bmin = 50
    #else:
    ax[ID].set_ylim(ymax=119.99)
    bmax = 19.99
    bmin = 100

    if ADD_BIOME:
        # biome patch bar -------------------------------

        
        # add color column
        df['color'] = [biome_color[i] for i in df.biome]

        # create biome patches
        patchlist = []

        # build patches
        for _, r in df.iterrows():
            p_ = mpatches.Rectangle((r['syear'], bmin), r['eyear']-r['syear'], bmax, ec="none", fc=r['color'], zorder=98)
            patchlist.append(p_)

        for p_ in patchlist:
            ax[ID].add_patch(p_)

        ax[ID].yaxis.get_ticklabels()[-1].set_visible(False)
        ax[ID].add_patch( mpatches.Rectangle((df.time.min(), bmin), df.time.max() - df.time.min(), bmax, ec="black", fc='none', zorder=99, linewidth=1))

        # add textmarker for cpation
        ax[ID].text(500, 101.5, '*', fontsize=13, zorder=100)



    # set some sensible runoff ylims
    if df.runoffavg.max() < 10:
        ax_right[ID].set_ylim(ymin=0, ymax=11.99)
    #elif df.runoffavg.max() < 100:
    #    ax_right[ID].set_ylim(ymin=0, ymax=109.99)
    elif df.runoffavg.max() < 550:
        ax_right[ID].set_ylim(ymin=0, ymax=599)
    else:
        ax_right[ID].set_ylim(ymin=400, ymax=1059)


    # panel now erosion rate
    ID += 1
    print(df.head())
    ax[ID].plot(df.time, df.erosion, linestyle='-', lw=0.75, color='black')
    ax[ID].set_ylabel(Dvar['erosion'], color='black')
    ax[ID].tick_params('y', colors=fg_color)
    ax[ID].axhline(y=0.2, linestyle='--', color='red', lw=1)
    ax[ID].set_ylim(ymin=0.0, ymax=0.5)



    # scale x axis on all panels ----------
    for ID in range(no_of_panels):
        ax[ID].get_yaxis().set_label_coords(-0.085,0.5)
        ax[ID].xaxis.grid(True, zorder=9999, linestyle='--')
        ax[ID].tick_params(labelsize=8)

        #if len(df) > 2000:
        ax[ID].set_xlim([0, 20000])
        ax[ID].set_xticks([0, 5000, 10000, 15000, 20000])
        
            #a=ax[ID].get_xticks().tolist()
            #labels = ["%d ka BP" % abs(x/1000) for x in a]
            #labels[-1] = 'PD'
            #labels[1] = ''
            #labels[3] = ''
            #labels[5] = ''
            #ax[ID].set_xticklabels( labels )

        # set tick and ticklabel color
        if ID > 0:
            ax[ID].axes.tick_params(color=fg_color, labelcolor=fg_color)
        ax[ID].set_facecolor(bg_color)
        for child in ax[ID].get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color(fg_color)

        
        if p in [0,2] and DROP_LABELS:
            # right
            #ax[ID].yaxis.set_major_formatter(plt.NullFormatter())
            ax[ID].yaxis.set_visible(False)

    
    # position y-axis labels
    for axr in ax_right.values():        
        axr.get_yaxis().set_label_coords(1.1,0.5)
        axr.set_facecolor(bg_color)
        axr.tick_params(labelsize=8)

        for child in axr.get_children():
            if isinstance(child, matplotlib.spines.Spine):
                child.set_color(fg_color)



        if p in [1,3] and DROP_LABELS:
            # right
            axr.yaxis.set_visible(False)
            #axr.yaxis.set_major_formatter(plt.NullFormatter())



    # LEGENDS ---------------------------------------

    leg_patches = []
    leg_names   = []
            
    # loop over pft groups
    
    D_pfts_site = OrderedDict()    
    for k, v in D_pfts.iteritems():
        x = []
        for i in range(len(v)):
            if v[i] in pft_names_site:
                x.append(v[i])
        if len(x)>0:
            D_pfts_site[k] = x

    for p_, pft_tuple in D_pfts_site.iteritems():
        pattern_right = None
        pattern_left = get_hatch( pft_tuple[0] )
        
        if len(pft_tuple)>1:
            pattern_right = get_hatch( pft_tuple[1] )
    
        l_item = AnyObject( D_cols2[p_], pattern_left, pattern_right ) 
        leg_patches.append( l_item )
        leg_names.append( p_ )
    
    # add the default legend item 
    leg_patches.append( AnyObject( 'lightgray', None, None) )
    leg_names.append( 'Various' )

    # ---------------------------
    PFT_PANEL_ID=2
    FPC_PANEL_ID=3

    do_pftlegend = True
    do_fpclegend = True
    
    if do_pftlegend:
        ncols=4
        if len(leg_patches) < 5:
            ncols = 2
        #if len(leg_patches) < 3:
        #    ncols = 1
        
        lpft = ax[PFT_PANEL_ID].legend(leg_patches, leg_names, 
                                #loc='center right', ncol=1, 
                                loc='upper left', ncol=ncols,
                                prop={'size':8}, frameon=True,
                                columnspacing=.6,
                                handler_map={AnyObject: AnyObjectHandler()})

        lpft.get_frame().set_linewidth(0.0)
        lpft.get_frame().set_facecolor(bg_color)
        for text in lpft.get_texts():
            text.set_color(fg_color)

        # activate to make legend appear to the right
        shift_to_right=False
        if shift_to_right:
            shift_legend(lpft, ax[PFT_PANEL_ID], 0.2, 0.1)


    if do_fpclegend:
        ncols=2
        leg_patches = [mpatches.Patch(color='darkgray'), 
                       mpatches.Patch(color='lightgray')]
        leg_names = ['Tree >5m', 'Other Veg']
        
        _pos = 'upper left'
        if df.Total.max() > 3:
            _pos = 'lower left'

        lpft = ax_right[FPC_PANEL_ID].legend(leg_patches , leg_names, 
                                loc=_pos, ncol=ncols,
                                prop={'size':8}, frameon=True,
                                columnspacing=.6)

        lpft.get_frame().set_linewidth(0.0)
        lpft.get_frame().set_facecolor(bg_color)
        for text in lpft.get_texts():
            text.set_color(fg_color)

        # activate to make legend appear to the right
        if _pos == 'upper left':
            shift_legend(lpft, ax[FPC_PANEL_ID], 0.0, 0.2)



    do_biomelegend = False
    if do_biomelegend:
    
        biomes_present = np.unique(df.biome)

        cs = []
        labs = []

        for b in biomes_present:
            cs.append( mpatches.Patch(color=D_cols_chile_es_new[b]) )
            labs.append( D_names_chile_es_new[b] )

        if len(biomes_present) > 3:
            ncols = 3
        else:
            ncols = len(biomes_present)

        lpft = ax[2].legend(cs, labs, loc='upper center', ncol=ncols, prop={'size':8} )

        lpft.get_frame().set_linewidth(0.0)
        lpft.get_frame().set_facecolor(bg_color)
        for text in lpft.get_texts():
            text.set_color(fg_color)



    foutname = fdataname[:-3]

    lf_id_str = ''
    if maxlfid:
        lf_id_str = '_lfid%d' % the_lf_id


    #plt.savefig('new2_dist100_plot_timeseries_focussites_pfts_%s_%s.png' % (clean(siteName[p]), bg_color), 
    plt.savefig(foutname + '_plot_ts_focussites_%d%s.png' % (p, lf_id_str) , #(clean(siteName[p])),
            bbox_inches='tight', dpi=300,
            facecolor=fig.get_facecolor(), edgecolor='none')


if __name__ == '__main__':
    #for p in range(3):
    #main(sys.argv[1], int(sys.argv[2]))

    maxlfid=False

    print("MAKE SURE TO FIX THE BIOMIZATION IMPORT BEFORE DOING THE PAPER!")

    fdataname = sys.argv[1]
    fbiomename = sys.argv[2]
    flandlab = sys.argv[3]
    site_id = int(sys.argv[4])
    if len(sys.argv)>4:
        maxlfid=True
        #if not 'lfid' in fdataname:
        #    print 'we need a lfid netCDF file'
        #    exit()

    main(fdataname, fbiomename, flandlab, site_id, maxlfid)
