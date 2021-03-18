#!/usr/bin/env python
# coding: utf-8

# # Import modules

import sys 
import string 
import subprocess
import copy
import netCDF4 as netcdf4 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy import interpolate
from scipy.stats import gamma, expon, norm
import xarray as xr
import xesmf as xe
import timeit
from os import path
import pandas as pd
# ---------------------------------------------------------- #
# Add geomorphic parameters and hillslope vegetation profiles
# to an existing surface data file
# ---------------------------------------------------------- #

import dask
from dask import delayed

from dask_jobqueue import PBSCluster
from dask.distributed import Client

cluster = PBSCluster(
    queue="regular",
    walltime="08:00:00",
    project="UCUB0089",
    memory="109GB",
    resource_spec="select=1:ncpus=36:mem=109GB",
    cores=36,
    processes=2,
)

#num_one_run = 60
num_worker = 60

# scale as needed
#cluster.adapt(minimum_jobs=18, maximum_jobs=30)
cluster.scale(60)

# Connect client to the remote dask workers
client = Client(cluster)

# Output directory
odir = '/glade/work/yifanc/NNA/script/generate_surfdata/nna4a/'          

# Specify hand data to use
anum = 2
if anum == 1:
    athresh = 10000
    annular_sf = 1
if anum == 2:
    athresh = 1000
    annular_sf = 5
if anum == 3:
    athresh = 200
    annular_sf = 10
    annular_sf = 2
if anum == 4:
    athresh = 50
    annular_sf = 20

nbins = 4

# if true, ignore DTND values greater than some threshold
removeTailDTND = True
#removeTailDTND = False

# if true, create 78pft surface data file, otherwise create 16pft file
doBGC = True
doBGC = False

# interpolate bedrock depth from valley to ridge
interpolateBedrockProfile = True
interpolateBedrockProfile = False

# use standard file and append new variables
snum = 1
if snum == 1:
    sfcfile = '/glade/work/yifanc/NNA/script/generate_surfdata/nna4a/surfdata_nna4a_hist_16pfts_Irrig_CMIP6_simyr2000_c210222.nc'
    outfile = odir + 'surfdata_nna4a_hist_16pfts_HAND_'+str(nbins)+'_col_hillslope_geo_params_'+str(athresh)+'_nlcd_annularx'+str(annular_sf)+'_bedrock_Yukon.nc'
            

if interpolateBedrockProfile:
    outfile = outfile.replace('bedrock_','bedrock_interp_')
    
print('base file is: ', sfcfile)

#efile0 = '/glade/scratch/swensosc/PYSHEDS/pysheds.nNullLatwNullLon_elv.geo_params_mask_flats_'+str(athresh)+'.nc'
geo_dir = '/glade/scratch/swensosc/PYSHEDS/'
print('geo params files dir: ',geo_dir)

veg_dir = '/glade/scratch/yifanc/GlobalLCT_Broxton/regrid_MODIS_LCT/'
print('land cover type file dir ',veg_dir)

dtb_dir = '/glade/scratch/swensosc/DTB/regridded_5d/'
print('depth to bedrock files dir: ',dtb_dir)

regrid_dir = '/glade/scratch/yifanc/CTSM_HH_Surfdata_mapping/'
print('directory of mapping files for geoparam /depth to bedrock files is: %s'%(regrid_dir))

ds_target = xr.open_dataset(sfcfile)
ds_target.load()

# generate grid boundary file
da_target_ = xr.DataArray(ds_target.LANDFRAC_PFT, dims=['x', 'y'],
                         coords = {'lon': (('x', 'y'), ds_target.LONGXY.values%360),
                                   'lat': (('x', 'y'), ds_target.LATIXY.values)})
# find the bounding box for each grid
target_lon = (da_target_.lon[0:-1,0:-1] + da_target_.lon[1:,0:-1] + da_target_.lon[0:-1,1:] + da_target_.lon[1:,1:])/4
target_lon_r_first = target_lon[:,0] - (target_lon[:,1] - target_lon[:,0])
target_lon_r_last = target_lon[:,-1] + (target_lon[:,-1] - target_lon[:,-2])
target_lon_update = np.concatenate([np.array([target_lon_r_first]).T,target_lon,np.array([target_lon_r_last]).T], axis=1)
target_lon_c_first = target_lon_update[0,:] - (target_lon_update[1,:] - target_lon_update[0,:])
target_lon_c_last = target_lon_update[-1,:] + (target_lon_update[-1,:] - target_lon_update[-2,:])
target_lon_update_2 = np.concatenate([np.array([target_lon_c_first]),target_lon_update,np.array([target_lon_c_last])], axis=0)

target_lat = (da_target_.lat[0:-1,0:-1] + da_target_.lat[1:,0:-1] + da_target_.lat[0:-1,1:] + da_target_.lat[1:,1:])/4
target_lat_r_first = target_lat[:,0] - (target_lat[:,1] - target_lat[:,0])
target_lat_r_last = target_lat[:,-1] + (target_lat[:,-1] - target_lat[:,-2])
target_lat_update = np.concatenate([np.array([target_lat_r_first]).T,target_lat,np.array([target_lat_r_last]).T], axis=1)
target_lat_c_first = target_lat_update[0,:] - (target_lat_update[1,:] - target_lat_update[0,:])
target_lat_c_last = target_lat_update[-1,:] + (target_lat_update[-1,:] - target_lat_update[-2,:])
target_lat_update_2 = np.concatenate([np.array([target_lat_c_first]),target_lat_update,np.array([target_lat_c_last])], axis=0)

# concat domain files
da_target_b_lon = xr.DataArray(target_lon_update_2, dims=['x_b', 'y_b'])
da_target_b_lat = xr.DataArray(target_lat_update_2, dims=['x_b', 'y_b'])
ds_target_ = xr.Dataset({'LAND':da_target_,'lon_b':da_target_b_lon, 'lat_b':da_target_b_lat})

# function definitions
def find_corresponding_latlon_id(lat,lon):
    '''
    find the corresponding lat/lon id for a selected lat/lon
    '''
    if lat>0:
        lat_pre='n'
    else:
        lat_pre='s'
    lat_val = "%i"%(np.floor(lat/5)*5)
    
    if lon>180:
        lon_pre='w'
        lon_val = "%i"%((np.floor(abs(lon-360)/5)+1)*5)
    else:
        lon_pre='e'
        lon_val = "%i"%(np.floor(lon/5)*5)
    return "%s%s%s%s"%(lat_pre,lat_val,lon_pre,lon_val)

def std_dev(x):
    return np.power(np.mean(np.power((x-np.mean(x)),2)),0.5)

def fit_polynomial(x,y,porder):
    tm = x.size
    # set up lsq matrix
    ncoefs = porder + 1
    g=np.zeros((tm,ncoefs))
    for n in range(ncoefs):
        if n == 0:
            g[:,n] = 1
        else:
            g[:,n]=np.power(x,n)

    gtd = np.dot(np.transpose(g), y)
    gtg = np.dot(np.transpose(g), g)

#  covm is the model covariance matrix
    covm = np.linalg.inv(gtg)
#  coefs is the model parameter vector
    coefs=np.dot(covm, gtd)

    return coefs

#--  synthesize polynomial to construct a x series
def synth_polynomial(x,coefs):
    tm = x.size
    ncoefs=coefs.size
    y=np.zeros((tm))
    for n in range(ncoefs):
        if n == 0:
            y += coefs[0]
        else:
            y += coefs[n] * np.power(x,n)
    return y

f = netcdf4.Dataset(sfcfile, 'r')
# coordinates
slon2d = np.asarray(f.variables['LONGXY'][:,])
slat2d = np.asarray(f.variables['LATIXY'][:,])
sjm,sim = np.shape(slon2d)
landmask = np.asarray(f.variables['PFTDATA_MASK'][:,])
pct_nat_pft = np.asarray(f.variables['PCT_NAT_PFT'][:,])
npft = pct_nat_pft.shape[0]
f.close()

# find dominant pft for each NLCD veg type
# 'vegetation type; grass=1, shrub=2, decid_tree=3, evergreen_tree=4, mixed_forest = 5'
num_nlcd_vegtype = 5
vegmap = np.zeros((num_nlcd_vegtype,sjm,sim),dtype=np.int)
for i in range(sim):
    for j in range(sjm):
        # grasses
        tmp = np.zeros((npft))
        tmp[12:15] = pct_nat_pft[12:15,j,i]
        vegmap[0,j,i] = np.argmax(tmp)
        # shrubs
        tmp = np.zeros((npft))
        tmp[9:12] = pct_nat_pft[9:12,j,i]
        vegmap[1,j,i] = np.argmax(tmp)
        # deciduous tree
        tmp = np.zeros((npft))
        for k in [3,6,7,8]:
            tmp[k] = pct_nat_pft[k,j,i]
        vegmap[2,j,i] = np.argmax(tmp)
        # evergreen tree
        tmp = np.zeros((npft))
        for k in [1,2,4,5]:
            tmp[k] = pct_nat_pft[k,j,i]
        vegmap[3,j,i] = np.argmax(tmp)
        # mixed forest; choose single pft
        tmp = np.zeros((npft))
        tmp[1:9] = pct_nat_pft[1:9,j,i]
        vegmap[4,j,i] = np.argmax(tmp)
            
naspect = 4 # N, E, S, W
aspect_bins = [[315,45],[45,135],[135,225],[225,315]]
dtr = np.pi/180.
    
ncolumns_per_gridcell = naspect * nbins
nhillslope = naspect

# for gridcells with no underlying DTB data
missing_bedrock_depth = 2.0

checkSinglePoint = True
checkSinglePoint = False
if checkSinglePoint:
    plotPDFs = True
    #plotPDFs = False
    
    plon = 244.75
    plat = 32.75
    plon = 245.25
    plat = 32.25
    plon = 239.75
    plat = 39.25
    plon = 238.75
    plat = 37.75

    plon = 238.25
    plat = 38.25
    #plon = 244.25
    #plat = 34.25
    plon = 242.25
    plat = 33.25
    #plon = 236.25
    #plat = 40.05

    istart = np.argmin(np.abs(slon-plon))
    jstart = np.argmin(np.abs(slat-plat))

    #istart = 15
    #jstart = 1
    print(jstart,istart)

    iend = istart+1
    jend = jstart+1
    printDebug = True
else:
    istart, iend = 0, sim
    jstart, jend = 0, sjm
    printDebug = False

# jstart=42 # for testing only
# jend=100   # for testing only
# Loop over points in CTSM domain

def find_subgrid(ijind = [j,i]):
    import xarray as xr
    import numpy as np
    import timeit
    col_cnt = 1
    time1 = timeit.default_timer()
    [j,i] = ijind
    # find target grid id (mapping purpose)
    target_grid_id = j * sim + i + 1

    # find latitude/longitude for the center of the grid cell
    lat_center = ds_target_.lat.values[j,i]
    lon_center = ds_target_.lon.values[j,i]

    # find corresponding 5deg grids for each target grid
    file_latlon_list = []
    for i_b in [i,i+1]:
        for j_b in [j,j+1]:
            lon_bound = da_target_b_lon.values[j_b,i_b]
            lat_bound = da_target_b_lat.values[j_b,i_b]
            file_latlon_list.append(find_corresponding_latlon_id(lat_bound,lon_bound))
    unique_file_latlon_list = np.unique(file_latlon_list)
    num_5deg_grid = len(unique_file_latlon_list)

    # read in data from digital elevation files
    shand = np.array([])
    sdtnd = np.array([])
    sarea = np.array([])
    sslope = np.array([])
    saspect = np.array([])
    sflood  = np.array([])
    svegtype = np.array([])
    sfrac_upland  = np.array([])
    sfrac_lowland = np.array([])
    suv_depth = np.array([])
    suh_depth = np.array([])
    slo_depth = np.array([])
    szmask    = np.array([])

    # get all sub-gridcell info within the target gridcell
    for latlon_id in unique_file_latlon_list:
        mapfile = regrid_dir + 'regridding_conservative.%s.nc'%(latlon_id)
        geofile = geo_dir + 'pysheds.%s_elv.geo_params_mask_flats_1000.nc'%(latlon_id)
        vegfile = veg_dir + 'MODIS_LCT_Broxton.%s.nc'%(latlon_id)
        dbtfile = dtb_dir + '5x5min_ORNL_SOILS_%s.nc'%(latlon_id)

        mf = xr.open_dataset(mapfile)
        gf = netcdf4.Dataset(geofile,'r')
        vf = netcdf4.Dataset(vegfile,'r')
        df = netcdf4.Dataset(dbtfile,'r')

        geo_lat_dim,geo_lon_dim=np.asarray(gf.variables['HAND'][:,]).shape

        geo_id = mf.col[mf.row==target_grid_id]
        if len(geo_id) > 0:
            y_ind_list = np.array((geo_id - 1)%geo_lon_dim, dtype=np.int32)
            x_ind_list = np.array(np.floor((geo_id - 1)/geo_lon_dim), dtype=np.int32)
#                    plt.scatter(gf.longitude[y_ind_list]%360,gf.latitude[x_ind_list], zorder=1, s=2)

            ihand = np.asarray(gf.variables['HAND'][:,])[x_ind_list,y_ind_list]
            idtnd = np.asarray(gf.variables['DTND'][:,])[x_ind_list,y_ind_list]
            iarea = np.asarray(gf.variables['area'][:,])[x_ind_list,y_ind_list]
            islope = np.asarray(gf.variables['slope'][:,])[x_ind_list,y_ind_list]
            iaspect = np.asarray(gf.variables['aspect'][:,])[x_ind_list,y_ind_list]
            iflood = np.asarray(gf.variables['flooded'][:,])[x_ind_list,y_ind_list]

            ivegtype = np.asarray(vf.variables['lct'][:,])[x_ind_list,y_ind_list]

            ifrac_upland = np.asarray(df.variables['frac_upland'][:,])[x_ind_list,y_ind_list] 
            ifrac_lowland = np.asarray(df.variables['frac_lowland'][:,])[x_ind_list,y_ind_list] 
            iuv_depth = np.asarray(df.variables['upland_valley_depth'][:,])[x_ind_list,y_ind_list]
            iuh_depth = np.asarray(df.variables['upland_hillslope_depth'][:,])[x_ind_list,y_ind_list] 
            ilo_depth = np.asarray(df.variables['lowland_depth'][:,])[x_ind_list,y_ind_list] 
            izmask = np.asarray(df.variables['mask'][:,])[x_ind_list,y_ind_list]

            shand = np.concatenate([shand,ihand])
            sdtnd = np.concatenate([sdtnd,idtnd])
            sarea = np.concatenate([sarea,iarea])
            sslope = np.concatenate([sslope,islope])
            saspect = np.concatenate([saspect,iaspect])
            sflood = np.concatenate([sflood,iflood])
            svegtype = np.concatenate([svegtype,ivegtype])
            sfrac_upland = np.concatenate([sfrac_upland,ifrac_upland])
            sfrac_lowland = np.concatenate([sfrac_lowland,ifrac_lowland])
            suv_depth = np.concatenate([suv_depth,iuv_depth])
            suh_depth = np.concatenate([suh_depth,iuh_depth])
            slo_depth = np.concatenate([slo_depth,ilo_depth])
            szmask = np.concatenate([szmask,izmask])
        gf.close()
        vf.close()
        df.close()
    # check for gridcells w/ no hand data
    # check for both all nans, or combination of zeros and nans
    s_df = pd.DataFrame(np.transpose([shand,sdtnd,sarea,sslope,saspect,sflood,svegtype,
                                      sfrac_upland,sfrac_lowland,suv_depth,suh_depth,slo_depth,szmask]),
                        columns=['shand','sdtnd','sarea','sslope','saspect','sflood','svegtype',
                                 'sfrac_upland','sfrac_lowland','suv_depth','suh_depth','slo_depth','szmask'])
    # initialize new fields to be added to surface data file
    vhand   = np.zeros((ncolumns_per_gridcell))
    vdtnd   = np.zeros((ncolumns_per_gridcell))
    varea   = np.zeros((ncolumns_per_gridcell))
    vslope  = np.zeros((ncolumns_per_gridcell))
    vaspect = np.zeros((ncolumns_per_gridcell))
    vwidth  = np.zeros((ncolumns_per_gridcell))
    vpftndx = np.zeros((ncolumns_per_gridcell))
    vzbedrock = np.zeros((ncolumns_per_gridcell))
    # length will not be added to file; it will be used to calculate width
    vlength  = np.zeros((ncolumns_per_gridcell))
    # indices begin with 1 (oceans are 0)
    vhillslope_index = np.zeros((ncolumns_per_gridcell))
    vcolumn_index    = np.zeros((ncolumns_per_gridcell))
    vdownhill_column_index  = np.zeros((ncolumns_per_gridcell))

    # remove nan values
    s_df = s_df.dropna(axis=0)

    if len(s_df) > 0:
        hand_coverage_fraction = s_df.shape[0]/shand.size
        hand_insufficient_data = (hand_coverage_fraction < 0.01)

#            if np.logical_or(hand_all_nans_check,hand_all_zeros_check):
    if len(s_df) == 0:
        print(slon2d[j,i],slat2d[j,i],'shand all nans or zeros')
    elif hand_insufficient_data:
        print('fraction of region hand > 0 {:10.6f}, skipping...'.format(hand_coverage_fraction))
    else:
        # continue processing data if some valid hand data exist
        if np.max(sdtnd) <= 0:
            print(slon2d[j,i],slat2d[j,i],'sdtnd all zeros')

        # remove values where hand is nan
        # this leads to hillslope area < gridcell area in some places
#                ind = np.where(np.isfinite(shand))[0]
        shand   = s_df["shand"].values   #shand[ind]
        sdtnd   = s_df["sdtnd"].values   #sdtnd[ind]
        sarea   = s_df["sarea"].values   #sarea[ind]
        sslope  = s_df["sslope"].values   #sslope[ind]
        saspect = s_df["saspect"].values   #saspect[ind]
        sflood  = s_df["sflood"].values   #sflood[ind]
        svegtype = s_df["svegtype"].values  #svegtype[ind]

        sfrac_upland  = s_df["sfrac_upland"].values    #sfrac_upland[ind]
        sfrac_lowland = s_df["sfrac_lowland"].values    #sfrac_lowland[ind]
        suv_depth = s_df["suv_depth"].values    #suv_depth[ind]
        suh_depth = s_df["suh_depth"].values    #suh_depth[ind]
        slo_depth = s_df["slo_depth"].values    #slo_depth[ind]
        szmask    = s_df["szmask"].values    #szmask[ind]
        del s_df
                        # eliminate tails of DTND distribution
        if removeTailDTND:
            std_dtnd = np.power(np.mean(np.power((sdtnd[shand > 0]-np.mean(sdtnd[shand > 0])),2)),0.5)
            fit_loc, fit_beta = expon.fit(sdtnd[shand > 0]/std_dtnd)
            rv = expon(loc=fit_loc,scale=fit_beta)
            npdf_bins = 5000
            pbins = np.linspace(0,np.max(sdtnd),npdf_bins)
            rvpdf = rv.pdf(pbins/std_dtnd)
            hval = 0.05
            r1 = np.argmin(np.abs(rvpdf - hval*np.max(rvpdf)))
            ind = np.where(sdtnd < pbins[r1])[0]

            if printDebug:
                print('{:d}% dtnd '.format(np.int(100*(1-hval))),np.max(sdtnd),pbins[r1])
            if ind.size > 0:
                shand   = shand[ind]
                sdtnd   = sdtnd[ind]
                sarea   = sarea[ind]
                sslope  = sslope[ind]
                saspect = saspect[ind]
                sflood  = sflood[ind]
                svegtype = svegtype[ind]

                sfrac_upland  = sfrac_upland[ind]
                sfrac_lowland = sfrac_lowland[ind]
                suv_depth = suv_depth[ind]
                suh_depth = suh_depth[ind]
                slo_depth = slo_depth[ind]
                szmask    = szmask[ind]


        # identify flooded regions in lowest hand bin
        hand_threshold = 2
        num_flooded_pts = np.sum((np.abs(sflood[shand < hand_threshold]) > 0))
        if num_flooded_pts > 0:
            # exclude regions that have been flooded
            # by eliminating 95% of flooded values
            flood_thresh = 0
            for ft in np.linspace(0,20,50):
                if (np.sum((np.abs(sflood[shand < hand_threshold]) > ft))/num_flooded_pts) < 0.95:
                    flood_thresh = ft
                    break

            shand = np.where(np.logical_and(np.abs(sflood) > flood_thresh,shand < hand_threshold),-1,shand)

        # give minimum value for dtnd
        smallest_dtnd = 1.0 # [meters]
        sdtnd[sdtnd <  smallest_dtnd] = smallest_dtnd

        if printDebug:
            print('total sarea: ',np.sum(sarea))

        # calculate hand bins that will give roughly equal areas
        # subject to maximum height of 2m for lowest bin
        # this could also be done by explicitly summing to get cdf
        std_hand = std_dev(shand[shand > 0])
        #fitHand = True
        fitHand = False
        if fitHand:
            fit_loc, fit_beta = expon.fit(shand[shand > 0].flat/std_hand)
            #x2m = 2./std_hand
            x2m = np.min([-fit_beta*np.log(1/4),2/std_hand])
            x33 = -fit_beta*np.log(2/3) + x2m
            x66 = -fit_beta*np.log(1/3) + x2m
            hand_bin_bounds = [0,x2m*std_hand,x33*std_hand,x66*std_hand,1e6]
        else:
            nhist = np.round(np.max(shand)).astype(np.int)
            bin1 = 2
            if nhist < bin1:
                print('max shand rounds to less than bin1')
                print(shand.size)
                print(slon2d[j,i],slat2d[j,i])
                #stop
                nhist = 200
                bin1 = 0

            hbins = np.linspace(bin1,np.max(shand),nhist+1)
            histo_hand = np.zeros((nhist))
            hbins_mid = 0.5*(hbins[:-1]+hbins[1:])
            for h in range(nhist):
                hind = np.where(np.logical_and(shand >= hbins[h],shand < hbins[h+1]))[0]
                histo_hand[h] = hind.size

            cum_histo_hand = np.zeros((nhist))
            for h in range(nhist):
                cum_histo_hand[h] = np.sum(histo_hand[:h+1])
            cum_histo_hand = cum_histo_hand/np.sum(histo_hand)

            # use quarters if bin1 = 0
            if bin1 == 0:
                b25  = hbins[np.argmin(np.abs(0.25 - cum_histo_hand))+1]
                b50  = hbins[np.argmin(np.abs(0.50 - cum_histo_hand))+1]
                b75  = hbins[np.argmin(np.abs(0.75 - cum_histo_hand))+1]
                hand_bin_bounds = [0,b25,b50,b75,1e6]
            else:
                b33  = hbins[np.argmin(np.abs(0.33 - cum_histo_hand))+1]
                b66  = hbins[np.argmin(np.abs(0.66 - cum_histo_hand))+1]
                hand_bin_bounds = [0,bin1,b33,b66,1e6]
                if b33 == b66:
                    #print('cumhist ',cum_histo_hand[0:10])
                    #print(hand_bin_bounds)
                    #print(slon[i],slat[j])
                    # just shift b66 for now
                    hand_bin_bounds = [0,bin1,b33,2*b33-bin1,1e6]

        if nbins != (len(hand_bin_bounds) - 1):
            print('bad hand bounds')
            stop

        if checkSinglePoint or printDebug:
            print('hand bin bounds ',hand_bin_bounds)

        # for each aspect, calculate hillslope elements
        for asp_ndx in range(naspect):
            if asp_ndx == 0:
                aind = np.where(np.logical_or(saspect >= aspect_bins[asp_ndx][0],saspect < aspect_bins[asp_ndx][1]))[0]
            else:
                aind = np.where(np.logical_and(saspect >= aspect_bins[asp_ndx][0],saspect < aspect_bins[asp_ndx][1]))[0]
            #aspect_coverage_fraction = aind.size/shand.size
            #aspect_insufficient_data = (aspect_coverage_fraction < 0.01)
            #print('aspect coverage fraction ',aspect_coverage_fraction)
            if aind.size > 0:
                # calculate geomorphic parameters in each bin
                determine_first_none_zero_col = 0
                for n in range(nbins):
                    b1 = hand_bin_bounds[n]
                    b2 = hand_bin_bounds[n+1]
                    hind = np.logical_and(shand[aind] >= b1,shand[aind] < b2)
                    cind = np.where(np.logical_and(sdtnd[aind] > 0,hind))[0]
                    if cind.size > 0:
                        cind = aind[cind]
                        vhand[asp_ndx*nbins+n] = np.mean(shand[cind])
                        vdtnd[asp_ndx*nbins+n] = np.mean(sdtnd[cind])
                        varea[asp_ndx*nbins+n] = np.sum(sarea[cind])
                        # exclude nans from calculation of mean slope
                        tmp = sslope[cind]
                        vslope[asp_ndx*nbins+n] = np.mean(tmp[np.isfinite(tmp)])

                        #if checkSinglePoint:
                        #    print('\narea ',n,area[asp_ndx*nbins+n,j,i])
                        #    print('\n')

                        '''
                        calculate along-profile length of each column
                        by fitting a gamma distribution to the
                        normalized dtnd values, then using the
                        peak or a fraction of maximum value as an
                        estimate of mean distnace
                        '''
                        hand_flat = shand[cind].flatten()
                        dtnd_flat = sdtnd[cind].flatten()
                        dtnd_std  = std_dev(dtnd_flat)

                        # if all distances equal
                        if dtnd_std == 0.0:
                            vlength[asp_ndx*nbins+n] = np.mean(dtnd_flat)
                        else:
                            fit_alpha, fit_loc, fit_beta = gamma.fit(dtnd_flat/dtnd_std)
                            # use exponential fit for alpha < 1
                            if fit_alpha < 1.0:
                                fit_loc, fit_beta = expon.fit(dtnd_flat/dtnd_std)
                                rv = expon(loc=fit_loc,scale=fit_beta)
                                if checkSinglePoint:
                                    print('expon fit ',fit_loc, fit_beta)
                            elif fit_alpha > 1e4:
                                # I had problems w/ gamma when wide spread in dtnd exists; switching to gaussian/normal distribution
                                fit_loc, fit_beta = norm.fit(dtnd_flat/dtnd_std)
                                rv = norm(loc=fit_loc,scale=fit_beta)
                                if checkSinglePoint:
                                    print('norm fit ',fit_loc, fit_beta)
                            else:
                                rv = gamma(fit_alpha,loc=fit_loc,scale=fit_beta)
                                if checkSinglePoint:
                                    print('gamma fit ',fit_alpha, fit_loc, fit_beta)

                            nhistobins = 500
                            nhistobins = 100
                            #hbins = np.linspace(0,np.max(dtnd_flat),nhistobins+1)
                            # keep bins the same for all aspects
#                            hbins = np.linspace(0,np.max(sdtnd[shand > 0]),nhistobins+1)
                            hbins = np.linspace(0,np.max(dtnd_flat[hand_flat > 0]),nhistobins+1)
                            hbins_mid = 0.5*(hbins[:-1]+hbins[1:])
                            rvpdf = rv.pdf(hbins_mid/dtnd_std)
                            if 1==1 and checkSinglePoint:
                                print('\ndtnd_std ',dtnd_std)
                                histo_dtnd = np.zeros((nhistobins))
                                for h in range(nhistobins):
                                    hind = np.where(np.logical_and(sdtnd[cind] >= hbins[h],sdtnd[cind] < hbins[h+1]))[0]
                                    histo_dtnd[h] = hind.size

                                m = np.dot(rvpdf,histo_dtnd)/np.dot(rvpdf,rvpdf)
                                cmap=copy.copy(plt.cm.jet)
                                if asp_ndx == 2:
                                    if plotPDFs:
                                        plt.plot(hbins_mid,histo_dtnd,marker='.',c=cmap(n/nbins),linewidth=0.5)
                                        plt.plot(hbins_mid,m*rvpdf,c=cmap(n/nbins))
                                        #plt.plot(hbins_mid,rvpdf)

                                if asp_ndx == -1:
                                    if plotPDFs:
                                        plt.plot(hbins_mid,rvpdf,linestyle='dashed')
                            max_rvpdf = np.max(rvpdf)
                            if checkSinglePoint:
                                print('max/loc pdf ',max_rvpdf,hbins_mid[np.argmax(rvpdf)])

                            # calculate location of shoulders
                            hval = 0.9
                            r1,r2 = rvpdf[:-1]-hval*max_rvpdf,rvpdf[1:]-hval*max_rvpdf
                            rind1 = np.where(np.logical_and(r1 <= 0,r2 >= 0))[0]
                            rind2 = np.where(np.logical_and(r1 >= 0,r2 <= 0))[0]
                            if rind1.size > 0:
                                w1 = hbins_mid[rind1[0]]
                            else:
                                w1 = 0
                            if rind2.size > 0:
                                w2 = hbins_mid[rind2[0]]
                            else:
                                w2 = 2*hbins_mid[np.argmax(rvpdf)] - w1

                            # rather than mean dtnd in bin,
                            # calc peak distance for gamma distributions, something greater than zero for exponential
                            if fit_alpha > 1:
                                vdtnd[asp_ndx*nbins+n] = hbins_mid[np.argmax(rvpdf)]
                            else:
                                vdtnd[asp_ndx*nbins+n] = w2
                                # avoid possible duplicate values
                                if n > 0 and (vdtnd[asp_ndx*nbins+n] == vdtnd[asp_ndx*nbins+n-1]):
                                    vdtnd[asp_ndx*nbins+n] = 2*w2

                            if checkSinglePoint:
                                print('\nw2/w1 ',n,(fit_alpha > 1),w2,w1)
                                #print('hbins ',hbins_mid[:6])

                            if w2-w1 <= 0:
                                print('\nbad length ',n,j,i,'----------')
                                print('lon/lat ',slon2d[j,i],slat2d[j,i])
                                print('w1,w2 ',w1,max_rvpdf,w2,hbins_mid[-1],np.max(dtnd_flat))

                        '''
                        aspect needs to be averaged using circular
                        (vector) mean rather than arithmatic mean
                        (to avoid cases of e.g. mean([355,5])->180,
                        when it should be 0)
                        '''

                        mean_aspect = np.arctan2(np.mean(np.sin(dtr*saspect[cind])),np.mean(np.cos(dtr*saspect[cind]))) / dtr
                        if mean_aspect < 0:
                            mean_aspect += 360.
                        vaspect[asp_ndx*nbins+n] = mean_aspect

                        if not np.isfinite(mean_aspect):
                            print('bad aspect: ',slon2d[j,i],slat2d[j,i],mean_aspect)

                        # calculate mean bedrock depth
                        zmask_tmp = szmask[cind]
                        tmp = sfrac_upland[cind]
                        ufrac = np.mean(tmp[zmask_tmp > 0])
                        tmp = sfrac_lowland[cind]
                        lfrac = np.mean(tmp[zmask_tmp > 0])
                        # compare upland/lowland fractions
                        if ufrac > lfrac:
                            if n == 0:
                                tmp = suv_depth[cind]
                            else:
                                tmp = suh_depth[cind]
                            vzbedrock[asp_ndx*nbins+n] = np.mean(tmp[zmask_tmp > 0])
                        else:
                            tmp = slo_depth[cind]
                            vzbedrock[asp_ndx*nbins+n] = np.mean(tmp[zmask_tmp > 0])
                        # if no bedrock data, set value
                        if np.max(zmask_tmp) == 0:
                            vzbedrock[asp_ndx*nbins+n] = missing_bedrock_depth

                        if checkSinglePoint:
                            print('\nfrac lowland/upland')
                            print(lfrac,ufrac)

                        # calculate vegetation type fractions in this bin
                        # ignore 0 (bare soil, but also crop etc)
                        cvegtype = svegtype[cind]
                        fgrass = np.sum(np.where(cvegtype == 1,1,0))
                        fshrub = np.sum(np.where(cvegtype == 2,1,0))
                        fdecid = np.sum(np.where(cvegtype == 3,1,0))
                        fevergreen = np.sum(np.where(cvegtype == 4,1,0))
                        fmixed = np.sum(np.where(cvegtype == 5,1,0))
                        fnorm = fgrass+fshrub+fdecid+fevergreen+fmixed
                        if fnorm > 0:
                            veg_frac = np.array([fgrass/fnorm,fshrub/fnorm,fdecid/fnorm,fevergreen/fnorm,fmixed/fnorm])
                        else:
                            veg_frac = np.array([1,0,0,0,0])
                        vndx = np.argmax(veg_frac)

                        #print('vegfrac ',j,i,veg_frac)

                        vpftndx[asp_ndx*nbins+n] = vegmap[vndx,j,i]

                        vhillslope_index[asp_ndx*nbins+n] = (asp_ndx+1)
                        vcolumn_index[asp_ndx*nbins+n] = col_cnt
                        if determine_first_none_zero_col == 0:
                            vdownhill_column_index[asp_ndx*nbins+n] = -9999
                            determine_first_none_zero_col += 1
                        else:
                            vdownhill_column_index[asp_ndx*nbins+n] = col_cnt -1
                        col_cnt += 1

                        if printDebug:
                            print('chk h/d/a: ',n,vhand[asp_ndx*nbins+n],vdtnd[asp_ndx*nbins+n],varea[asp_ndx*nbins+n])

                # check that lowermost bin is less distant than neighboring bin
                # this can occur when a gridcell contains both a plain and
                # a hilly/mountainous region
                if vdtnd[asp_ndx*nbins]-vdtnd[asp_ndx*nbins+1] >= 0:
                    print('\nlowest bin DTND too large')
                    print('lon/lat ',slon2d[j,i],slat2d[j,i])
                    print(vdtnd[asp_ndx*nbins:asp_ndx*nbins+nbins],'\n')

                    if np.logical_and(vdtnd[asp_ndx*nbins] == 0.0,vdtnd[asp_ndx*nbins+1] == 0.0):
                        #    stop
                        break

                    x = np.log(vhand[asp_ndx*nbins:(asp_ndx+1)*nbins])
                    y = np.log(vdtnd[asp_ndx*nbins:(asp_ndx+1)*nbins])
                    # y=ax^b, linear in log(x)
                    pcoefs = fit_polynomial(x[1:],y[1:],1)
                    vdtnd[asp_ndx*nbins] = np.exp(synth_polynomial(x[0],pcoefs))
                    # set minimum value for distance
                    vdtnd[asp_ndx*nbins] = np.max([1.0,vdtnd[asp_ndx*nbins]])

                # if bin distances overlap, combine bins
                '''
                combine bins, set n column_index to 0
                average hand and slope
                combine area and width
                leave dtnd and aspect unchanged
                adjust indices to account for removal
                '''
                n = 1
                while n < nbins:
                    if (vdtnd[asp_ndx*nbins+n-1] >= vdtnd[(asp_ndx*nbins+n):(asp_ndx*nbins+nbins)]).any():
                        m = n
                        while (vdtnd[asp_ndx*nbins+n-1] >= vdtnd[asp_ndx*nbins+m]) and (m<nbins):                   
                            print('dtnd not monotonically increasing ', asp_ndx)
                            print(vdtnd[asp_ndx*nbins+0:asp_ndx*nbins+nbins])
                            print('combining columns ', n-1,' and ',m)
                            print('lon/lat ',slon2d[j,i],slat2d[j,i],'\n')

                            mean_hand = (vhand[asp_ndx*nbins+n-1]*varea[asp_ndx*nbins+n-1] + vhand[asp_ndx*nbins+m]*varea[asp_ndx*nbins+m])/(varea[asp_ndx*nbins+n-1] + varea[asp_ndx*nbins+m])
                            vhand[asp_ndx*nbins+n-1] = mean_hand
                            mean_slope = (vslope[asp_ndx*nbins+n-1]*varea[asp_ndx*nbins+n-1] + vslope[asp_ndx*nbins+m]*varea[asp_ndx*nbins+m])/(varea[asp_ndx*nbins+n-1] + varea[asp_ndx*nbins+m])
                            vslope[asp_ndx*nbins+n-1] = mean_slope

                            varea[asp_ndx*nbins+n-1] += varea[asp_ndx*nbins+m]
                            vwidth[asp_ndx*nbins+n-1] += vwidth[asp_ndx*nbins+m]
                            mean_bedrock = (vzbedrock[asp_ndx*nbins+n-1]*varea[asp_ndx*nbins+n-1] + vzbedrock[asp_ndx*nbins+m]*varea[asp_ndx*nbins+m])/(varea[asp_ndx*nbins+n-1] + varea[asp_ndx*nbins+m])
                            vzbedrock[asp_ndx*nbins+n-1] = mean_bedrock
                            print("------------------only for test: ",mean_hand, mean_slope, mean_bedrock, asp_ndx, n)
                            # set removed column values to zero
                            vhand[asp_ndx*nbins+m]   = 0
                            vdtnd[asp_ndx*nbins+m]   = 0
                            vslope[asp_ndx*nbins+m]  = 0
                            vaspect[asp_ndx*nbins+m] = 0
                            varea[asp_ndx*nbins+m]   = 0
                            vwidth[asp_ndx*nbins+m]  = 0
                            vzbedrock[asp_ndx*nbins+m]  = 0
                            vcolumn_index[asp_ndx*nbins+m] = 0
                            # decrement indices to account for removal
                            vcolumn_index[asp_ndx*nbins+n+1:] -= 1
                            vdownhill_column_index[asp_ndx*nbins+n+1:] -= 1
                            col_cnt -= 1
                            m += 1
                        n = m
                    else:
                        n += 1

                # interpolate bedrock depth from valley to ridge
                if interpolateBedrockProfile:
                    n_min_slope = np.argmin(vslope[asp_ndx*nbins:(asp_ndx+1)*nbins])
                    n_max_slope = np.argmax(vslope[asp_ndx*nbins:(asp_ndx+1)*nbins])
                    #zm = (zbedrock[asp_ndx*nbins+n_min_slope,j,i] - zbedrock[asp_ndx*nbins+n_max_slope,j,i])/(slope[asp_ndx*nbins+n_min_slope,j,i] - slope[asp_ndx*nbins+n_max_slope,j,i])
                    # assume zero slope for valley center
                    zm = (vzbedrock[asp_ndx*nbins+n_min_slope] - vzbedrock[asp_ndx*nbins+n_max_slope])/(0 - vslope[asp_ndx*nbins+n_max_slope])
                    zb = vzbedrock[asp_ndx*nbins+n_max_slope] - zm * vslope[asp_ndx*nbins+n_max_slope]
                    vzbedrock[asp_ndx*nbins:(asp_ndx+1)*nbins] = zm*vslope[asp_ndx*nbins:(asp_ndx+1)*nbins] + zb

                # base widths on assumption of annular rings
                # starting w/ uppermost bin
                # add scalar to account for not circularity
                #for n in range(nbins-1,-1,-1):
                for n in range(nbins):
                    vwidth[asp_ndx*nbins+n] = annular_sf * 2*np.pi*np.sqrt(np.sum(varea[asp_ndx*nbins+n:asp_ndx*nbins+nbins])/np.pi)
                    # double width of lowest column
                    if n == 0:
                        vwidth[asp_ndx*nbins+n] = 2*vwidth[asp_ndx*nbins+n]
                    if checkSinglePoint:
                        print('areas ',n,nbins-1,varea[asp_ndx*nbins+n:asp_ndx*nbins+nbins])
                        print('width ',n,vwidth[asp_ndx*nbins+n])
                        if vwidth[asp_ndx*nbins+nbins-1] > 0:
                            print('width ratio ',vwidth[asp_ndx*nbins+n]/vwidth[asp_ndx*nbins+nbins-1])

            if checkSinglePoint:
                print('\n---------  aspect', asp_ndx+1,'----------')
                print('area_all_columns ', np.sum(varea[:ind.size]),np.sum(varea[:]))
                print('width ',vwidth[asp_ndx*nbins:asp_ndx*nbins+nbins])
                print('distance ',vdtnd[asp_ndx*nbins:asp_ndx*nbins+nbins])
                print('area ',varea[asp_ndx*nbins:asp_ndx*nbins+nbins])
                print('bedrock ',vzbedrock[asp_ndx*nbins:asp_ndx*nbins+nbins])
                print('max distance ', vdtnd[asp_ndx*nbins+nbins-1])

    v_df = pd.DataFrame(np.transpose([vhand,vdtnd,varea,vslope,vaspect,vwidth,vpftndx,
                          vzbedrock,vlength,vhillslope_index,vcolumn_index,vdownhill_column_index]),
            columns=['vhand','vdtnd','varea','vslope','vaspect','vwidth','vpftndx',
                     'vzbedrock','vlength','vhillslope_index',
                     'vcolumn_index','vdownhill_column_index'])
    # remove nan values
#    s_df = s_df.dropna(axis=0)
    temp_out_dir = "/glade/scratch/yifanc/CTSM_HH_Surfdata_mapping/temp_file/"
    v_df.to_csv(temp_out_dir + 'subgrid_files.params.j_%s_i_%s.csv'%(j,i),index=False)

    time2 = timeit.default_timer()
    print("It takes %s seconds to process j=%s i=%s"%(time2-time1,j,i))
    return 1

import dask.bag as db
count = 0
pair_list = []
temp_out_dir = "/glade/scratch/yifanc/CTSM_HH_Surfdata_mapping/temp_file/"
for i in range(sim):
    for j in range(sjm):
        filename = temp_out_dir + 'subgrid_files.params.j_%s_i_%s.csv'%(j,i)
        if (landmask[j,i] >0) and (not path.exists(filename)): 
            count += 1
            pair_list.append([j,i])
print(count)
b = db.from_sequence(pair_list, npartitions=num_worker)
b = b.map(find_subgrid)
b.compute()
