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

# Output directory
odir = '/glade/work/yifanc/NNA/optmz/input/surfdata/nna4a/updated_hand_bins/'

# Specify hand data to use
'''
anum = 3
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
'''

dic_aspect_bins = {2:[[270,90],[90,270]],
                   4:[[315,45],[45,135],[135,225],[225,315]]}

sathresh = sys.argv[1]
sannular_sf = sys.argv[2]
snbins = sys.argv[3]
snaspect = sys.argv[4] # N, E, S, W

athresh = int(sathresh)
annular_sf = int(sannular_sf)
nbins = int(snbins)
naspect = int(snaspect)

aspect_bins = dic_aspect_bins[naspect] #[[315,45],[45,135],[135,225],[225,315]]
dtr = np.pi/180.
ncolumns_per_gridcell = naspect * nbins
nhillslope = naspect

# define the threshold of hillslope fraction
# below which the hillslope will be merged with nearby 
# hillslopes
thres_pct_hs_frac = 2.0
thres_pct_c_frac = 0.5

# define the grid-level parameter directory
temp_out_dir = "/glade/scratch/yifanc/CTSM_HH_Surfdata_mapping/delineation_thres_200/temp_file/athres_%s_annular_%s_nbins_%s_naspect_%s/"%(sathresh,sannular_sf,snbins,snaspect)
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
    outfile = odir + 'surfdata_nna4a_hist_16pfts_HAND_'+str(nbins)+'_aspect_'+str(naspect)+'_col_hillslope_geo_params_'+str(athresh)+'_nlcd_annularx'+str(annular_sf)+'_bedrock_pe.merge_tiny_hillslope.nc'
            

if interpolateBedrockProfile:
    outfile = outfile.replace('bedrock_','bedrock_interp_')
    
print('base file is: ', sfcfile)

domain_ds = xr.open_dataset('/glade/work/yifanc/NNA/optmz/input/domain.arctic.pe.4km.c210630.nc')
landmask = domain_ds.mask.values


# define function
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

# define the domain file
f = netcdf4.Dataset(sfcfile, 'r')
# coordinates
slon2d = np.asarray(f.variables['LONGXY'][:,])
slat2d = np.asarray(f.variables['LATIXY'][:,])
sjm,sim = np.shape(slon2d)
pct_nat_pft = np.asarray(f.variables['PCT_NAT_PFT'][:,])
npft = pct_nat_pft.shape[0]
f.close()

# initialize new fields to be added to surface data file
hand   = np.zeros((ncolumns_per_gridcell,sjm,sim))
dtnd   = np.zeros((ncolumns_per_gridcell,sjm,sim))
area   = np.zeros((ncolumns_per_gridcell,sjm,sim))
slope  = np.zeros((ncolumns_per_gridcell,sjm,sim))
aspect = np.zeros((ncolumns_per_gridcell,sjm,sim))
width  = np.zeros((ncolumns_per_gridcell,sjm,sim))
pftndx = np.zeros((ncolumns_per_gridcell,sjm,sim))
zbedrock = np.zeros((ncolumns_per_gridcell,sjm,sim))
# length will not be added to file; it will be used to calculate width
length  = np.zeros((ncolumns_per_gridcell,sjm,sim))

# copy pct_nat_pft for cases of no hand data
wpct_nat_pft = np.copy(pct_nat_pft)

pct_hillslope   = np.zeros((nhillslope,sjm,sim))
# indices begin with 1 (oceans are 0)
hillslope_index = np.zeros((ncolumns_per_gridcell,sjm,sim))
column_index    = np.zeros((ncolumns_per_gridcell,sjm,sim))
downhill_column_index  = np.zeros((ncolumns_per_gridcell,sjm,sim))

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

# Loop over points in CTSM domain
time1 = timeit.default_timer()
count = 0
for j in range(jstart,jend):
    for i in range(istart,iend):
        if landmask[j,i] > 0:            
            filename = temp_out_dir + 'subgrid_files.params.j_%s_i_%s.csv'%(j,i)
            df = pd.read_csv(filename)
            hand[:,j,i] = df['vhand'].values
            dtnd[:,j,i] = df['vdtnd'].values
            area[:,j,i] = df['varea'].values
            slope[:,j,i] = df['vslope'].values
            aspect[:,j,i] = df['vaspect'].values
            width[:,j,i] = df['vwidth'].values
            pftndx[:,j,i] = df['vpftndx'].values
            zbedrock[:,j,i] = df['vzbedrock'].values
            length[:,j,i] = df['vlength'].values
            hillslope_index[:,j,i] = df['vhillslope_index'].values
            column_index[:,j,i] = df['vcolumn_index'].values
            downhill_column_index[:,j,i] = df['vdownhill_column_index'].values
            count += 1
time2 = timeit.default_timer()
print('it takes %s seconds to process %s grids'%(time2-time1, count))

# modify pct_nat_pft when pft was missing 
for j in range(jstart,jend):
    for i in range(istart,iend):
        if landmask[j,i] > 0:
            hillslope_ind_sel = hillslope_index[:,j,i]
            ind = np.where(hillslope_ind_sel>0)[0]
            pct_pft_sel = pct_nat_pft[:,j,i]
            h_pftndx_sel = pftndx[:,j,i][ind]
            num_pftndx = len(np.unique(h_pftndx_sel))
            num_overlap = len(set(np.where(pct_pft_sel>0)[0]).intersection(np.unique(h_pftndx_sel)))
            if num_pftndx != num_overlap:
                print('We will assign 1.0% to each missing pft')
                pft_missing = list(set(np.unique(h_pftndx_sel)).difference(np.where(pct_pft_sel>0)[0]))
                num_pft_missing = len(pft_missing)
                print('Missing pft: %s'%(pft_missing))
                pft_w_largest_fraction = np.argmax(pct_pft_sel)
                pct_pft_w_largest_fraction = pct_pft_sel[pft_w_largest_fraction]
                print('PCT with largest fraction: %s, pct: %s'%(pft_w_largest_fraction, pct_pft_w_largest_fraction))
                pct_pft_w_largest_fraction_adjusted = pct_pft_w_largest_fraction - float(num_pft_missing)
                print('Adjust the pft with largest pct')
                wpct_nat_pft[pft_w_largest_fraction,j,i] = pct_pft_w_largest_fraction_adjusted
                for pft in pft_missing:
                    wpct_nat_pft[int(pft),j,i] = 1.0
                print('Adjusted nat_pft_pct:%s'%(wpct_nat_pft[:,j,i]), np.sum(wpct_nat_pft[:,j,i]))

# Compress data along column axis
nhillcolumns = np.zeros((sjm,sim))
for j in range(jstart,jend):
    for i in range(istart,iend):
        if landmask[j,i] > 0:
            ind = np.where(column_index[:,j,i] > 0)[0]
            if checkSinglePoint:
                print('\ncompressing data')
                print('hand ', hand[ind,j,i])
                print('dtnd ', dtnd[ind,j,i])
                print('slope ', slope[ind,j,i])
                print('bedrock ', zbedrock[ind,j,i])
                print('pftndx ', pftndx[ind,j,i])
                print('col_ndx ',column_index[ind,j,i])
                #print('dcol_ndx ', downhill_column_index[ind,j,i])
            nhillcolumns[j,i] = ind.size
            hand[:ind.size,j,i]   = hand[ind,j,i]
            dtnd[:ind.size,j,i]   = dtnd[ind,j,i]
            area[:ind.size,j,i]   = area[ind,j,i]
            slope[:ind.size,j,i]  = slope[ind,j,i]
            aspect[:ind.size,j,i] = aspect[ind,j,i]
            width[:ind.size,j,i]  = width[ind,j,i]
            zbedrock[:ind.size,j,i]  = zbedrock[ind,j,i]
            pftndx[:ind.size,j,i] = pftndx[ind,j,i]
            hillslope_index[:ind.size,j,i] = hillslope_index[ind,j,i]
            column_index[:ind.size,j,i]    = column_index[ind,j,i]
            downhill_column_index[:ind.size,j,i] = downhill_column_index[ind,j,i]

            harea = area[:ind.size,j,i]
            area_all_columns = np.sum(harea)
            hndx = hillslope_index[:ind.size,j,i]
            if area_all_columns > 0:
                for n in range(naspect):
                    area_hillslope = np.sum(harea[hndx == (n+1)])
                    pct_hillslope[n,j,i] = 100*(area_hillslope/area_all_columns)

#            ia = np.argmin(np.abs(slon-240.25))
#            ja = np.argmin(np.abs(slat-32.75))
#            if np.logical_and(i==ia,j==ja):
#                print('pcthill ',ja,ia,pct_hillslope[:,ja,ia])
                    
            # set unused portion of ncolumns to zero
            if ind.size < ncolumns_per_gridcell:
                hand[ind.size:,j,i] = 0
                dtnd[ind.size:,j,i] = 0
                area[ind.size:,j,i] = 0
                slope[ind.size:,j,i] = 0
                aspect[ind.size:,j,i] = 0
                width[ind.size:,j,i] = 0
                zbedrock[ind.size:,j,i] = 0
                pftndx[ind.size:,j,i] = 0
                hillslope_index[ind.size:,j,i] = 0 
                column_index[ind.size:,j,i] = 0
                downhill_column_index[ind.size:,j,i] = 0

# Check that all points in land mask have valid data
# if not, fill from nearest neighbor
if not checkSinglePoint:
    for j in range(jstart,jend):
        for i in range(istart,iend):
            if landmask[j,i] > 0:
                if np.all(column_index[:,j,i] == 0):
                    tmp = np.ma.array(np.abs(slon2d-slon2d[j,i])+np.abs(slat2d-slat2d[j,i]),mask=(nhillcolumns < 1))
                    j1, i1 = np.unravel_index(tmp.argmin(),tmp.shape)
                    #j1, i1 = np.unravel_index(np.argmin(tmp),tmp.shape)

                    print('pt ',j,i)
                    print('npt ',j1,i1)
                    print('filling point with value from neighbor')
                    print('nhillcolumns: ',nhillcolumns[j1,i1])
                    mask=(nhillcolumns < 1) ; print(mask[j1,i1])
                    print(slon2d[j,i],slat2d[j,i],slon2d[j1,i1],slat2d[j1,i1])
                    print('nh ',nhillcolumns[j,i],nhillcolumns[j1,i1])
                    print('pcthill ',pct_hillslope[:,j,i],pct_hillslope[:,j1,i1])
                    print('area ',area[:,j1,i1])

                    nhillcolumns[j,i] = nhillcolumns[j1,i1]
                    hand[:,j,i] = hand[:,j1,i1]
                    dtnd[:,j,i] = dtnd[:,j1,i1]
                    area[:,j,i] = area[:,j1,i1]
                    slope[:,j,i] = slope[:,j1,i1]
                    aspect[:,j,i] = aspect[:,j1,i1]
                    width[:,j,i] = width[:,j1,i1]
                    zbedrock[:,j,i] = zbedrock[:,j1,i1]
                    # pftndx needs to have nonzero pct_nat_pft
                    wpct_nat_pft[:,j,i] = wpct_nat_pft[:,j1,i1]

                    #print('-----------------------------')

                    #print('pct pft: ',pct_nat_pft[pftndx[:,j,i].astype(np.int32),j,i])
                    #print('pct pft: ',pct_nat_pft[:,j,i])
                    #print('new pftndx: ',pftndx[:,j1,i1])

                    pftndx[:,j,i] = pftndx[:,j1,i1]

                    hillslope_index[:,j,i] = hillslope_index[:,j1,i1]
                    column_offset = (np.max(column_index) +1 - column_index[0,j1,i1])
                    column_index[:,j,i] = column_index[:,j1,i1] + column_offset
                    downhill_column_index[:,j,i] = downhill_column_index[:,j1,i1] + column_offset
                    pct_hillslope[:,j,i] = pct_hillslope[:,j1,i1]

#stop
    
                
# if any dtnd out of order or equal, fit a polynomial and adjust distances
for j in range(jstart,jend):
    for i in range(istart,iend):
        if landmask[j,i] > 0:
            for hndx in range(1,naspect+1):
                ind = np.where(hillslope_index[:,j,i] == hndx)[0]
                delta_hand = hand[ind[1:],j,i] - hand[ind[0:-1],j,i]
                delta_dtnd = dtnd[ind[1:],j,i] - dtnd[ind[0:-1],j,i]
                # fit a power law (y=ax^b)
                if np.any(delta_dtnd <= 0):
                    print('dtnd out of order or equal ')
                    print('lon/lat ',slon2d[j,i],slat2d[j,i])
                    if checkSinglePoint:
                        #plt.plot(dtnd[ind,j,i],hand[ind,j,i],marker='o',c='b')
                        pass
                    x = np.log(hand[ind,j,i])
                    y = np.log(dtnd[ind,j,i])
                    pcoefs = fit_polynomial(x,y,1)
                    # synth returns logy not y
                    dtnd[ind,j,i] = np.exp(synth_polynomial(x,pcoefs))
                    # are there any negative dtnd?
                    negind = np.where(dtnd[ind,j,i] < 0)[0]
                    if negind.size > 0:
                        dtnd[ind[negind],j,i] = 1

                    if checkSinglePoint:
                        #plt.plot(dtnd[ind,j,i],hand[ind,j,i],marker='o',c='r')
                        print('poly dtnd ',hndx,dtnd[ind,j,i])
                        
# check that no negative values occurred when fitting polynomial                
x = np.min(hand)
print('min hand after poly: ',x)

# check that dtnd not degenerate
#if np.logical_and(not hand_all_nans_check, not hand_all_zeros_check) and not hand_insufficient_data:
for j in range(jstart,jend):
    for i in range(istart,iend):
        if landmask[j,i] > 0:
            for hndx in range(1,naspect+1):
                ind = np.where(hillslope_index[:,j,i] == hndx)[0]
                delta_dtnd = dtnd[ind[1:],j,i] - dtnd[ind[0:-1],j,i]
                if np.any(delta_dtnd == 0):
                    print('\ndegenerate dtnd')
                    print(dtnd[ind,j,i])
                    print(slon2d[j,i],slat2d[j,i])
                    print('nh ',nhillcolumns[j,i])


# merge hillslope with areal fraction smaller than threshold

# ======== Functions for merging ========= #
def merge_hillslopes(pct_hs,aspect_c,hillslope_index_c,hand_c):
    '''
    we will merge the hillslope with smallest 
    areal fraction
    this function will output 
    1. dictionary for {ID for the deleted hillslope:ID for the target hillslope}
    2. dictionary for {ID for the deleted columns:ID for the target columns}
    '''
    # number of hillsloep and columns
    pct_nonzero = pct_hs[pct_hs>0]
    num_c = len(np.unique(hillslope_index_c[hillslope_index_c>0]))
    num_hs = len(np.unique(pct_nonzero))
    pct_min = np.min(pct_nonzero)
    hillslope_id_del = np.where(pct_hs == pct_min)[0][0] + 1 # the id start from 1
    col_index_del = np.where(hillslope_index_c == hillslope_id_del)[0]
    aspect_hs_del = np.arctan2(np.mean(np.sin(dtr*aspect_c[col_index_del])),
                               np.mean(np.cos(dtr*aspect_c[col_index_del]))) / dtr
    # calculate the average aspect for each hillslope
    direction_difference = []
    hs_remaining_list = []
    for hs in range(1,num_hs+1):
        if (hs != hillslope_id_del) and (hs in hillslope_index_c):
            cind = np.where(hillslope_index_c == hs)[0]
            aspect_hs = np.arctan2(np.mean(np.sin(dtr*aspect_c[cind])),np.mean(np.cos(dtr*aspect_c[cind]))) / dtr
            if aspect_hs < 0:
                aspect_hs += 360.
            aspect_dot_unit = np.dot([np.sin(dtr*aspect_hs),np.cos(dtr*aspect_hs)],
                                     [np.sin(dtr*aspect_hs_del),np.cos(dtr*aspect_hs_del)])
            hs_remaining_list.append(hs)
            direction_difference.append(aspect_dot_unit)

    # find the nearest hillslope to merge (nearest aspect)
    hs_remaining_list = np.array(hs_remaining_list).astype(np.int64)
    hillslope_id_target = hs_remaining_list[direction_difference==np.max(direction_difference)][0]
    col_index_target = np.where(hillslope_index_c == hillslope_id_target)[0]
    # for each columns in the hillslope, find which columns to merge
    c_from_to = {}
    for col_del in col_index_del:
        dist_del_target_list = []
        for col_target in col_index_target:
            dist_del_target_list.append(abs(hand_c[col_del]-hand_c[col_target]))
        print(col_del,col_target,dist_del_target_list)
        c_target_id = col_index_target[dist_del_target_list == np.min(dist_del_target_list)][0]
        c_from_to.update({col_del:c_target_id})
    hs_from_to = {hillslope_id_del:hillslope_id_target}
    return c_from_to, hs_from_to

def update_merging_add_list(data, dict_merge, scenario):
    '''
    scenario: whether the ID starts from 0 or start from 1
    '''
    for del_id in dict_merge.keys():
        target_id = dict_merge[del_id]
        data[target_id-scenario] += data[del_id-scenario]
        data[del_id-scenario] = 0
    # find non-zero data
    data_output = np.zeros(len(data))
    data_non_zero = data[data>0]
    data_output[0:len(data_non_zero)] = data_non_zero
    return data_output

def update_merging_weighted_average_list(data, weight_data, dict_merge, scenario, whether_aspect=False):
    '''
    scenario: whether the ID starts from 0 or start from 1
    '''
    for del_id in dict_merge.keys():
        target_id = dict_merge[del_id]
        w_d = weight_data[del_id-scenario]
        w_t = weight_data[target_id-scenario]
        norm_d = w_d/(w_d+w_t)
        norm_t = w_t/(w_d+w_t)
        t_id = target_id-scenario
        d_id = del_id-scenario
        if not whether_aspect:
            data[t_id] = data[t_id]*norm_t + data[d_id]*norm_d
        else:
            data[t_id] = np.arctan2((np.sin(dtr*data[t_id])*norm_t+np.sin(dtr*data[d_id])*norm_d),
                                    (np.cos(dtr*data[t_id])*norm_t+np.cos(dtr*data[d_id])*norm_d)) / dtr
            if data[t_id] < 0:
                data[t_id] += 360.
        data[d_id] = 0

    # find non-zero data
    data_output = np.zeros(len(data))
    data_non_zero = data[data>0]
    data_output[0:len(data_non_zero)] = data_non_zero
    return data_output

def update_hillslope_index(data, dict_merge):
    for i_del in dict_merge.keys():
        data[data==i_del] = 0
        data[data>i_del] -= 1
    # find non-zero data
    data_output = np.zeros(len(data))
    data_non_zero = data[data>0]
    data_output[0:len(data_non_zero)] = data_non_zero
    return data_output

def update_downhill_hillslope_index(data, dict_merge):
    for i_del in dict_merge.keys():
        data[i_del] = 0
        data[i_del:] -= 1
        print(data)
    # find non-zero data
    data_output = np.zeros(len(data))
    data_non_zero = data[(data>0)|(data<=-9999)]
    data_output[0:len(data_non_zero)] = data_non_zero
    data_output[data_output<-9999] = -9999
    data_output[(data_output>-9999)&(data_output<0)] = 0
    return data_output

def update_pftndx(data, dict_merge):
    for i_del in dict_merge.keys():
        data[i_del] = 0
    # find non-zero data
    data_output = np.zeros(len(data))
    data_non_zero = data[data>0]
    data_output[0:len(data_non_zero)] = data_non_zero
    return data_output

for j in range(jstart,jend):
    for i in range(istart,iend):
        if landmask[j,i] > 0:
            pct_hs = pct_hillslope[:,j,i]
            num_c = int(nhillcolumns[j,i])
            num_hs = int(np.sum(np.unique(hillslope_index[:,j,i])>0))
            if (pct_hs[pct_hs>0] < thres_pct_hs_frac).any():
                while (pct_hs[pct_hs>0] < thres_pct_hs_frac).any():
                    print("start processing j=%s, i=%s"%(j,i))
                    dict_c_from_to, dict_hs_from_to = merge_hillslopes(pct_hs,aspect[:,j,i],hillslope_index[:,j,i],hand[:,j,i])
                    num_hs -= len(dict_hs_from_to)
                    num_c -= len(dict_c_from_to)
                    area_c = area[:,j,i]
                    pct_hillslope[:,j,i] = update_merging_add_list(pct_hs,dict_hs_from_to,1)
                    hand[:,j,i] = update_merging_weighted_average_list(hand[:,j,i],area_c,dict_c_from_to,0)
                    dtnd[:,j,i] = update_merging_weighted_average_list(dtnd[:,j,i],area_c,dict_c_from_to,0)
                    width[:,j,i] = update_merging_weighted_average_list(width[:,j,i],area_c,dict_c_from_to,0)
                    slope[:,j,i] = update_merging_weighted_average_list(slope[:,j,i],area_c,dict_c_from_to,0)
                    aspect[:,j,i] = update_merging_weighted_average_list(aspect[:,j,i],area_c,dict_c_from_to,0,True)
                    zbedrock[:,j,i] = update_merging_weighted_average_list(zbedrock[:,j,i],area_c,dict_c_from_to,0)
                    pftndx[:,j,i] = update_pftndx(pftndx[:,j,i],dict_c_from_to)
                    nhillcolumns[j,i] = num_c
                    hillslope_index[:,j,i] = update_hillslope_index(hillslope_index[:,j,i],dict_hs_from_to)
                    column_index[num_c:,j,i] = 0
                    downhill_column_index[:,j,i] = update_downhill_hillslope_index(downhill_column_index[:,j,i], dict_c_from_to)
                    #  update in area must be in the very last
                    area[:,j,i] = update_merging_add_list(area[:,j,i],dict_c_from_to,0)
                    pct_hs = pct_hillslope[:,j,i]

def merge_columns(area_c,hillslope_index_c,hand_c):
    '''
    This function merges the columns with tiny areal fraction.
    Please note the column can only be merged with columns within this hillslope
    '''
    # step 1: find columns to be deleted
    pct_area_c = area_c/np.sum(area_c) * 100
    min_pct_area_c = np.min(pct_area_c[pct_area_c>0]) # we only find the columns with non-zero area
    c_del_ind = np.where(pct_area_c == min_pct_area_c)[0][0]
    # step 2: find target columns
    hs_del = hillslope_index_c[c_del_ind]
    c_hs = np.where(hillslope_index_c == hs_del)[0]
    hand_diff_list = []
    c_sel_list = []
    for c_sel in c_hs:
        if c_sel != c_del_ind:
            hand_diff_list.append(abs(hand_c[c_del_ind]-hand_c[c_sel]))
            c_sel_list.append(c_sel)
    c_sel_list = np.array(c_sel_list).astype(np.int64)
    c_target_ind = c_sel_list[hand_diff_list == np.min(hand_diff_list)][0]    
    return {c_del_ind:c_target_ind}

def update_downhill_hillslope_index_c(data, dict_merge, hillslope_index_c):
    for i_del in dict_merge.keys():
        data[i_del] = 0
        hillslope_index_del = hillslope_index_c[i_del]
        c_within_hs = np.where(hillslope_index_c == hillslope_index_del)[0]
        if c_within_hs[0] == i_del:
            data[c_within_hs[1]] = -9999
        else:
            data[c_within_hs[0]] = -9999
        data[i_del:] -= 1

    # find non-zero data
    data_output = np.zeros(len(data))
    data_non_zero = data[(data>0)|(data<=-9999)]
    data_output[0:len(data_non_zero)] = data_non_zero
    data_output[data_output<-9999] = -9999
    data_output[(data_output>-9999)&(data_output<0)] = 0
    return data_output

def update_hillslope_index_c(data, dict_merge):
    for i_del in dict_merge.keys():
        data[i_del] = 0    
    # find non-zero data
    data_output = np.zeros(len(data))
    data_non_zero = data[data>0]
    data_output[0:len(data_non_zero)] = data_non_zero
    return data_output

# merge columns with small areal fractions
for j in range(jstart,jend):
    for i in range(istart,iend):
        if landmask[j,i] > 0:
            area_c = area[:,j,i]
            pct_area_c = area_c/np.sum(area_c) * 100
            num_c = int(nhillcolumns[j,i])
            if (pct_area_c[pct_area_c>0] < thres_pct_c_frac).any():
                while (pct_area_c[pct_area_c>0] < thres_pct_c_frac).any():
                    dict_c_from_to = merge_columns(area_c,hillslope_index[:,j,i],hand[:,j,i])
                    num_c -= len(dict_c_from_to)
                    hand[:,j,i] = update_merging_weighted_average_list(hand[:,j,i],area_c,dict_c_from_to,0)
                    dtnd[:,j,i] = update_merging_weighted_average_list(dtnd[:,j,i],area_c,dict_c_from_to,0)
                    width[:,j,i] = update_merging_weighted_average_list(width[:,j,i],area_c,dict_c_from_to,0)
                    slope[:,j,i] = update_merging_weighted_average_list(slope[:,j,i],area_c,dict_c_from_to,0)
                    aspect[:,j,i] = update_merging_weighted_average_list(aspect[:,j,i],area_c,dict_c_from_to,0,True)
                    zbedrock[:,j,i] = update_merging_weighted_average_list(zbedrock[:,j,i],area_c,dict_c_from_to,0)
                    pftndx[:,j,i] = update_pftndx(pftndx[:,j,i],dict_c_from_to)
                    nhillcolumns[j,i] = num_c
                    # update downhill_column_index needs to happen before updating hillslope_index
                    downhill_column_index[:,j,i] = update_downhill_hillslope_index_c(downhill_column_index[:,j,i], dict_c_from_to,hillslope_index[:,j,i])
                    hillslope_index[:,j,i] = update_hillslope_index_c(hillslope_index[:,j,i],dict_c_from_to)
                    column_index[num_c:,j,i] = 0
                    #  update in area must be in the very last
                    area[:,j,i] = update_merging_add_list(area[:,j,i],dict_c_from_to,0)
                    area_c = area[:,j,i]
                    pct_area_c = area_c/np.sum(area_c) * 100

if checkSinglePoint:
    plt.show()
    stop

# output file
da_hand = xr.DataArray(hand,dims=['nmaxhillcol','lsmlat','lsmlon'])
da_dtnd = xr.DataArray(dtnd,dims=['nmaxhillcol','lsmlat','lsmlon'])
da_width = xr.DataArray(width,dims=['nmaxhillcol','lsmlat','lsmlon'])
da_area = xr.DataArray(area,dims=['nmaxhillcol','lsmlat','lsmlon'])
da_slope = xr.DataArray(slope,dims=['nmaxhillcol','lsmlat','lsmlon'])
da_aspect = xr.DataArray(aspect,dims=['nmaxhillcol','lsmlat','lsmlon'])
da_bedrock = xr.DataArray(zbedrock,dims=['nmaxhillcol','lsmlat','lsmlon'])
da_pftndx = xr.DataArray(pftndx,dims=['nmaxhillcol','lsmlat','lsmlon'])
da_nhillcolumns = xr.DataArray(nhillcolumns,dims=['lsmlat','lsmlon'])
da_pct_hillslope = xr.DataArray(pct_hillslope,dims=['nhillslope','lsmlat','lsmlon'])
da_hillndx = xr.DataArray(hillslope_index,dims=['nmaxhillcol','lsmlat','lsmlon'])
da_column_index = xr.DataArray(column_index,dims=['nmaxhillcol','lsmlat','lsmlon'])
da_downhill_column_index = xr.DataArray(downhill_column_index,dims=['nmaxhillcol','lsmlat','lsmlon'])

# define data type
da_hand = da_hand.astype(dtype=np.float64)
da_dtnd = da_dtnd.astype(dtype=np.float64)
da_width = da_width.astype(dtype=np.float64)
da_area = da_area.astype(dtype=np.float64)
da_slope = da_slope.astype(dtype=np.float64)
da_aspect = da_aspect.astype(dtype=np.float64)
da_bedrock = da_bedrock.astype(dtype=np.float64)
da_pftndx = da_pftndx.astype(dtype=np.int32)
da_nhillcolumns = da_nhillcolumns.astype(dtype=np.int32)
da_pct_hillslope = da_pct_hillslope.astype(dtype=np.float64)
da_hillndx = da_hillndx.astype(dtype=np.int32)
da_column_index = da_column_index.astype(dtype=np.int32)
da_downhill_column_index = da_downhill_column_index.astype(dtype=np.int32)

# define attrs
da_hand.attrs = {'units':'m','long_name':'hillslope height'}
da_dtnd.attrs = {'units':'m','long_name':'hillslope length'}
da_width.attrs = {'units':'m','long_name':'hillslope width'}
da_area.attrs = {'units':'m2','long_name':'hillslope area'}
da_slope.attrs = {'units':'m/m','long_name':'hillslope slope'}
da_aspect.attrs = {'units':'radians','long_name':'hillslope aspect (clockwise from North)'}
da_bedrock.attrs = {'units':'m','long_name':'hillslope bedrock depth'}
da_pftndx.attrs = {'units':'m','long_name':'hillslope pft indices'}
da_nhillcolumns.attrs = {'units':'unitless','long_name':'number of columns per landunit'}
da_pct_hillslope.attrs = {'units':'per cent','long_name':'percent hillslope of landunit'}
da_hillndx.attrs = {'units':'unitless','long_name':'hillslope_index'}
da_column_index.attrs = {'units':'unitless','long_name':'column index'}
da_downhill_column_index.attrs = {'units':'unitless','long_name':'downhill column index'}

# attach to output files
ds_target['h_height']=da_hand
ds_target['h_length']=da_dtnd
ds_target['h_width']=da_width
ds_target['h_area']=da_area
ds_target['h_slope']=da_slope
ds_target['h_aspect']=da_aspect
ds_target['h_bedrock']=da_bedrock
ds_target['h_pftndx']=da_pftndx
ds_target['nhillcolumns']=da_nhillcolumns
ds_target['pct_hillslope']=da_pct_hillslope
ds_target['hillslope_index']=da_hillndx
ds_target['column_index']=da_column_index
ds_target['downhill_column_index']=da_downhill_column_index

# add _FillValue information to encoding
encoding_dic = {}
for var in ds_target.variables:
    encoding_dic.update({var:{'_FillValue': None}})

# output
ds_target.to_netcdf(outfile, encoding=encoding_dic, format='NETCDF4_CLASSIC')


'''
# Write data to file

command='date "+%y%m%d"'
timetag=subprocess.Popen(command,stdout=subprocess.PIPE,shell='True').communicate()[0].strip().decode()

# copy original file
command=['cp',sfcfile,outfile]
x=subprocess.call(command,stderr=subprocess.PIPE)

w = netcdf4.Dataset(outfile, 'a')
w.creation_date = timetag
if sfcfile != None:
    w.input_file = sfcfile

#w.createDimension('lsmlon',sim)
#w.createDimension('lsmlat',sjm)
w.createDimension('nhillslope',nhillslope)
w.createDimension('nmaxhillcol',ncolumns_per_gridcell)

# for high resolution dems, use 64bit precision for coordinates
ohand = w.createVariable('h_height',np.float64,('nmaxhillcol','lsmlat','lsmlon',))
ohand.units = 'm'
ohand.long_name = 'hillslope height'

odtnd = w.createVariable('h_length',np.float64,('nmaxhillcol','lsmlat','lsmlon',))
odtnd.units = 'm'
odtnd.long_name = 'hillslope length'

owidth = w.createVariable('h_width',np.float64,('nmaxhillcol','lsmlat','lsmlon',))
owidth.units = 'm'
owidth.long_name = 'hillslope width'

oarea = w.createVariable('h_area',np.float64,('nmaxhillcol','lsmlat','lsmlon',))
oarea.units = 'm2'
oarea.long_name = 'hillslope area'

oslop = w.createVariable('h_slope',np.float64,('nmaxhillcol','lsmlat','lsmlon',))
oslop.units = 'm/m'
oslop.long_name = 'hillslope slope'

oasp  = w.createVariable('h_aspect',np.float64,('nmaxhillcol','lsmlat','lsmlon',))
oasp.units = 'radians'
oasp.long_name = 'hillslope aspect (clockwise from North)'

obed = w.createVariable('h_bedrock',np.float64,('nmaxhillcol','lsmlat','lsmlon',))
obed.units = 'meters'
obed.long_name = 'hillslope bedrock depth'

opft = w.createVariable('h_pftndx',np.int32,('nmaxhillcol','lsmlat','lsmlon',))
opft.units = 'unitless'
opft.long_name = 'hillslope pft indices'

onhill = w.createVariable('nhillcolumns',np.int32,('lsmlat','lsmlon',))
onhill.units = 'unitless'
onhill.long_name = 'number of columns per landunit'

opcthill  = w.createVariable('pct_hillslope',np.float64,('nhillslope','lsmlat','lsmlon',))
opcthill.units = 'per cent'
opcthill.long_name = 'percent hillslope of landunit'

ohillndx  = w.createVariable('hillslope_index',np.int32,('nmaxhillcol','lsmlat','lsmlon',))
ohillndx.units = 'unitless'
ohillndx.long_name = 'hillslope_index'

ocolndx  = w.createVariable('column_index',np.int32,('nmaxhillcol','lsmlat','lsmlon',))
ocolndx.units = 'unitless'
ocolndx.long_name = 'column index'

odcolndx  = w.createVariable('downhill_column_index',np.int32,('nmaxhillcol','lsmlat','lsmlon',))
odcolndx.units = 'unitless'
odcolndx.long_name = 'downhill column index'

ohand[:,]  = hand
odtnd[:,]  = dtnd
oarea[:,]  = area
owidth[:,] = width
oslop[:,]  = slope
obed[:,]   = zbedrock
# aspect should be in radians on surface data file
oasp[:,]   = aspect * dtr
opcthill[:,] = pct_hillslope
opft[:,]     = pftndx.astype(np.int32)
onhill[:,]   = nhillcolumns.astype(np.int32)
ohillndx[:,] = hillslope_index.astype(np.int32)
ocolndx[:,]  = column_index.astype(np.int32)
odcolndx[:,] = downhill_column_index.astype(np.int32)

w.variables['PCT_NAT_PFT'][:,] = wpct_nat_pft[:,]

w.close()
print(outfile+' created')
'''
