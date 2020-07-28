# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:52:02 2020

@author: rober
"""


import os, numpy as np, csv, matplotlib.pyplot as plt, scipy.optimize as opt, math, struct, binascii, gc, time, random
import sys
from operator import sub
from joblib import Parallel, delayed
import scipy#, lmfit
from scipy.optimize import minimize # used for implementation of maximum likelihood exponential fit
from matplotlib import gridspec
import matplotlib.colors as mcolors
from math import factorial
from math import *
from scipy.stats import poisson
get_ipython().run_line_magic('matplotlib', 'auto')
import matplotlib as mpl
import pickle
import numba as nb
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib as mpl
#mpl.rcParams['figure.dpi']= 300
import socket #enables to find computer name to make less of a mess with folders
# sys.path.append('C:/Users/rober/Documents/Doktorat/GitLab/apd_ple')
# import Read_PLE_functions as rpl

sys.path.append('E:/Martijn/ETH/scripts/github/martijn_own_github/repoo/apd_ple')
import Read_PLE_functions as rpl
sys.path.append('E:/Martijn/ETH/scripts/github/martijn_own_github/repoo/martijn_own_github')
import Read_PLE_functions_Martijn as rplm
InVoltagenew_c=nb.jit(nopython=True)(rplm.InVoltagenew)
#%%
basefolders={'DESKTOP-BK4HAII':'C:/Users/rober/Documents/Doktorat/Projects/SingleParticle_PLE/Andor Spectrometer/',
             'HP_Probook':'E:/Martijn/ETH/results/',
             'mavt-omel-w004w':'E:/LAB_DATA/Robert/'} #dictionary which base folder to use on which computer. For you this probably would be everything until 
folder = basefolders[socket.gethostname()]+'20200612_HR148_RT/'

filename = 'HR148_4p87MHz_200Hz_500mVpp_60mVoffset_QD2_ND1'

settingsfile= filename+'_settings'
HHsettings={}
HHsettings=rpl.load_obj(settingsfile, folder )
# HHsettings["syncRate"]=4866880
Texp = 240 # total time [s] of measurements
binwidth = 0.01 # width of time bins to create intensity trace [s]

# nrmeas = len(namelist) #nr of data files
# nrtbins = int(Texp/binwidth) # nr of bins in intensity trace
# Ilimslist = np.ndarray(shape=(nrmeas,nrbins),dtype=float)
# Alphabinlist = np.full((nrmeas,nrbins),0,dtype=float) # init list to save Alpha of each intensity bin
# Thetabinlist = np.full((nrmeas,nrbins),0,dtype=float) # init list to save Alpha of each intensity bin
# Abinlist = np.full((nrmeas,nrbins),0,dtype=float) # init list to save maximum histogram amplitudes (at t=t0)
# taubinlist = np.zeros(nrtbins-1,dtype=float) # init list to save lifetimes of each time bin (binwidth)
# intbinlist = np.zeros(nrtbins-1,dtype=float) # init list to save intensity per bin

data = rpl.ImportT3(folder + filename + '.out',HHsettings)
Texp = round(data[8]*data[1]*1024,0) # macrotime values can store up to 2^10 = 1024 before overflow. overflows*1024*dtmacro gives experiment time [s]
print('averaged cps on det0 and det1:',np.array(data[6:8])/Texp)
print('experimental time in s:',Texp)
[microtimes0,microtimes1,times0,times1,dtmicro,dtmacro,decaytlist,decayylist] = rpl.ShiftPulsedData(data[2],data[4],data[3],data[5],data[0],data[1]) #decaytlist and decayylist are the two variables you want to check for the modulation trace
# MaxLikelihoodFunction_c = nb.jit(nopython=True)(rpl.MaxLikelihoodFunction)
lifetime=rpl.GetLifetime(microtimes0,dtmicro,dtmacro,dtfit=400e-9,tstart=-1,histbinmultiplier=1,ybg=0,plotbool=True,method='ML_c',expterms=2)

limits0 = rpl.HistPhotons(times0*dtmicro,binwidth,Texp) #gives index of first photon of det0 in each bin
limits1 = rpl.HistPhotons(times1*dtmicro,binwidth,Texp)

# make an intensity trace and find Imax
inttrace = rpl.MakeIntTrace(limits0,limits1,binwidth,Texp)

#%% PLE specific
Voltage=np.array([-80,-40,0,40,80])
Wavelength_calib=np.array([578.6,560.2,541.5,560.3,504.9])
calibcoeffs=np.polyfit(Voltage,Wavelength_calib,1)
# Vpp=240
# Freq=200
# Voffset=0
Vpp=float(filename.split('mVpp_')[0].split('_')[-1])
Freq=float(filename.split('Hz_')[-2])
Voffset=float(filename.split('mVoff')[0].split('_')[-1])
tshift=rplm.Findtshift(Freq,Vpp,Voffset,calibcoeffs,data[19],dtmacro,(480,570),(-4e-4,-2e-4),Debugmode=True,histbinnumber=200) #coarse sweep
tshift=rplm.Findtshift(Freq,Vpp,Voffset,calibcoeffs,data[19],dtmacro,(480,570),(tshift-1e-5,tshift+1e-5),Debugmode=True,histbinnumber=200)
reffolder = folder

calibspec=rpl.importASC(reffolder+'backref_9p734MHz_200Hz_500mVpp_60mVoffset.asc')
cmaxpos=np.argmax(np.mean(calibspec[0][:,:],1))
refspecold=np.mean(calibspec[0][cmaxpos-5:cmaxpos+5,:],0)

refspec=np.mean(calibspec[0][cmaxpos-5:cmaxpos+5,:],0)-np.mean(refspecold[900:len(refspecold)])
refspec=refspec/(1240/calibspec[1])
refspec=refspec/np.max(refspec)
# refspec=savgol_filter(refspec, 23, 3)
interprefspec= interp1d(calibspec[1], refspec,kind='cubic',fill_value='extrapolate')
plt.figure()
plt.plot(np.linspace(500,600,300),interprefspec(np.linspace(500,600,300)))
plt.plot(calibspec[1],refspec)

#%% Microtime resolved PLE
plt.figure()
maxtime=100

mtimelims=np.array([0,100])
binning=1
# timebins=np.logspace(np.log10(1),np.log10(np.max(data[2]*data[0]*1e9)-lifetime[2]-1),10)
# timebins=np.linspace(0,150)
timebins=rpl.Easyhist(data[2]*data[0]*1e9,0,200*dtmicro*1e9,dtmicro*1e9)[2]
# timebins=np.logspace(np.log10(1),np.log10(150),50)
wavelbins=np.linspace(480,550,70)

mtimelimsdtmic=np.round(mtimelims/data[0]*1e-9)
nbins=int(mtimelimsdtmic[1]-mtimelimsdtmic[0])
lambdalim=[500,600]
plt.hist2d(data[2]*data[0]*1e9,calibcoeffs[1]+rpl.InVoltagenew(data[19]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],range=[[mtimelimsdtmic[0]*data[0]*1e9,mtimelimsdtmic[1]*data[0]*1e9],lambdalim],bins=[(nbins-1)*binning,50],norm=mcolors.LogNorm())
#plt.hist2d(data[2]*data[0]*1e9-lifetime[2]-1,calibcoeffs[1]+InVoltagenew(data[19]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],bins=[timebins,wavelbins])#,norm=mcolors.LogNorm())
Histdata,xedges,yedges=np.histogram2d(data[2]*data[0]*1e9,calibcoeffs[1]+rpl.InVoltagenew(data[19]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],bins=[timebins,wavelbins])
# plt.xlabel('Time (ns)')
# plt.ylabel('Excitation wavelength (nm)')
# plt.colorbar()
#%% PLE spectrum
histsettings=(420,610,0.2)
exspec=rpl.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(data[19]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
plt.plot(exspec[0],exspec[1])
#%% Find peak in time for each wavelength
# maxtimes=np.zeros(len(wavelbins)-1)
# wavels=np.zeros(len(wavelbins)-1)
# for k in range(len(wavelbins)-1):
#     maxtimes[k]=np.argmax(Histdata[:,k])
#     wavels[k]=wavelbins[k]
# plt.figure()
# plt.plot(wavels[:58],maxtimes[:58])
# starttime=np.polyfit(wavels[:58],maxtimes[:58],1)
# plt.plot(wavels,starttime[1]+wavels*starttime[0])
#%% Macrotime-resolved PLE



timebins=np.linspace(0,int(Texp),int(Texp*5))
wavelbins=np.linspace(430,600,200)
Histdata,xedges,yedges=np.histogram2d(data[3]*data[1],calibcoeffs[1]+rpl.InVoltagenew(data[19]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],bins=[timebins,wavelbins])#,norm=mcolors.LogNorm())
X, Y = np.meshgrid(timebins, wavelbins)
plt.pcolormesh(X, Y, Histdata.T)
# plt.xlim(data[26][0]*dtmacro,data[26][-1]*dtmacro)
plt.xlim(0,Texp)
plt.xlabel('Time (s)')
plt.ylabel('Excitation/Emission wavelength')




#%% Load PL
emisspec=rpl.importASC(reffolder+filename+'.asc')
# cmaxpos=np.argmax(np.mean(np.mean(linespec[0],2)[:,400:600],1))
# refspec=np.mean(calibspec[0][cmaxpos-5:cmaxpos+5,:],0)-np.mean(refspecold[925:len(refspecold)])


# ysize=3
# emisspecold=np.sum(emisspec[0][cmaxpos-ysize:cmaxpos+ysize,:,:],0)
emisspecold=emisspec[0][0,:,:]
# emisphotons=(emisspecold-473*(2*ysize))*5.36/emisspec[3]['Gain']
emisphotons=(emisspecold-473)*5.36/emisspec[3]['Gain']

#%% Excitation spectra over each other
timerangelist=np.array([[8.3,18],[110,111]])
# timerangelist=np.array([[44,384],[391,403],[405,420],[455,463]])
timerange=np.array([timerangelist[0],timerangelist[1]])
histsettings=(430,630,1)
# k=4
ntimepoints=2
fig,((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
for k in range(ntimepoints):

    limitphotons=(np.argmin(np.abs(times0*dtmicro-timerange[k][0])),np.argmin(np.abs(times0*dtmicro-timerange[k][1])))
    exspec=rpl.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(data[19][limitphotons[0]:limitphotons[1]]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
    # plt.plot(exspec[0],exspec[1]/interprefspec(exspec[0])/(timerange[k][1]-timerange[k][0])+50*k)
    # ax1.plot(exspec[0],exspec[1]/interprefspec(exspec[0])/(limitphotons[1]-limitphotons[0])+0.002*k,label=str(timerange[k][0])+' to '+str(timerange[k][1])+' s')
    ax1.plot(exspec[0],exspec[1]/(limitphotons[1]-limitphotons[0]),label=str(timerange[k][0])+' to '+str(timerange[k][1])+' s')
secax = ax1.secondary_xaxis('top', functions=(rpl.eVtonm,rpl.nmtoeV))
secax.set_xlabel('Energy (eV)')
ax1.set_xlabel('Wavelength (nm)')
# ax1.legend()
ax1.set_ylabel('PLE')
ax1.set_yticks([])


# emwavelrange=(595,630)
# for k in range(ntimepoints):
#     limitframes=(np.argmin(np.abs(data[26]*dtmacro-timerange[k][0])),np.argmin(np.abs(data[26]*dtmacro-timerange[k][1])))
#     emspec=np.sum(emisphotons[:,limitframes[0]:limitframes[1]],1)
#     # plt.plot(emisspec[1][emwavelindices[0]:emwavelindices[1]],emspec[emwavelindices[0]:emwavelindices[1]]/(timerange[k][1]-timerange[k][0])+50*k)
#     ax2.plot(emisspec[1][emwavelindices[0]:emwavelindices[1]],emspec[emwavelindices[0]:emwavelindices[1]]/np.sum(emspec[emwavelindices[0]:emwavelindices[1]]),label=str(timerange[k][0])+' to '+str(timerange[k][1])+' s')
# secax = ax2.secondary_xaxis('top', functions=(rpl.eVtonm,rpl.nmtoeV))
# secax.set_xlabel('Energy (eV)')
# ax2.set_xlabel('Wavelength (nm)')
# ax2.set_yticks([])
# ax2.set_ylabel('Emission Intensity')
# # ax2.legend()
# plt.tight_layout()

# emwavelindices=(np.argmin(np.abs(emisspec[1]-emwavelrange[0])),np.argmin(np.abs(emisspec[1]-emwavelrange[1])))
# X, Y = np.meshgrid(data[26]*dtmacro, emisspec[1][emwavelindices[0]:emwavelindices[1]])
# plt.figure()
# ax3.pcolormesh(X, Y, emisphotons[emwavelindices[0]:emwavelindices[1],:len(data[26])])
# plt.imshow(emisphotons)
# plt.colorbar()

timebins=np.linspace(0,int(Texp),int(Texp))
wavelbins=np.linspace(520,630,120)
Histdata,xedges,yedges=np.histogram2d(data[3]*data[1],calibcoeffs[1]+rpl.InVoltagenew(data[19]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],bins=[timebins,wavelbins])#,norm=mcolors.LogNorm())
X, Y = np.meshgrid(timebins, wavelbins)
ax3.pcolormesh(X, Y, Histdata.T)
# plt.xlim(data[26][0]*dtmacro,data[26][-1]*dtmacro)
# plt.xlim(120,250)

# for k in range(ntimepoints):
    # ax3.plot(np.array([timerange[k][0],timerange[k][0],timerange[k][1],timerange[k][1],timerange[k][0]]),np.array([histsettings[0]+5,emwavelrange[1]-5,emwavelrange[1]-5,histsettings[0]+5,histsettings[0]+5]),'--')
# ax3.set_xlabel('Time (s)')
# ax3.set_ylabel('Excitation/Emission wavelength')


maxexwavel=600
for k in range(ntimepoints):
    # k=0
    limitphotons=(np.argmin(np.abs(times0*dtmicro-timerange[k][0])),np.argmin(np.abs(times0*dtmicro-timerange[k][1])))
    indicesinrange=np.argwhere((calibcoeffs[1]+rpl.InVoltagenew(data[19][limitphotons[0]:limitphotons[1]]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0]<maxexwavel))
    timetrace=rpl.Easyhist(microtimes0[limitphotons[0]:limitphotons[1]][indicesinrange]*dtmicro,0,dtmicro*10000,dtmicro*4)
    # plt.figure()
    # ax3.semilogy((timetrace[0])*1e9,timetrace[1],label=str(timerange[k][0])+' to '+str(timerange[k][1])+' s')
    # plt.semilogy((timetrace[0])*1e9,2e1*np.exp(-(timetrace[0])*1e9/(115)))
    # plt.axes(ax4)
    rpl.GetLifetime(microtimes0[limitphotons[0]:limitphotons[1]][indicesinrange],dtmicro,dtmacro,dtfit=150e-9,tstart=-1,histbinmultiplier=1,ybg=(timerange[k][1]-timerange[k][0])*60/floor(dtmacro/dtmicro),plotbool=True,method='ML_c',expterms=1)
    # ax3.set_xlim(0,220)
    # plt.ylim([])
# ax3.legend()
ax4.set_xlabel('Time (ns)')
ax4.set_ylabel('Intensity')
# plt.xlabel('Wavelength (nm)')
# rpl.GetLifetime(microtimes0[limitphotons[0]:limitphotons[1]],dtmicro,dtmacro,dtfit=100e-9,tstart=50e-9,histbinmultiplier=1,ybg=0,plotbool=True,method='ML_c')
# limitphotons=(np.argmin(np.abs(times0*dtmicro-timerange[k][0])),np.argmin(np.abs(times0*dtmicro-timerange[k][1])))
# rpl.GetLifetime(microtimes0[limitphotons[0]:limitphotons[1]],dtmicro,dtmacro,dtfit=5e-9,tstart=-1,histbinmultiplier=1,ybg=0,plotbool=True,method='ML_c')


#%% Lifetime vs Intensity
# MaxLikelihoodFunction_c = nb.jit(nopython=True)(MaxLikelihoodFunction)
macrotimesin=data[3]
# macrotimesin=data[25]
microtimesin=data[2]
#microtimesin=microtimesblue
#macrotimesin=macrotimesblue
#macrotimescyclein=macrotimescycleblue
binwidth=0.01
macrolimits = rpl.HistPhotons(macrotimesin*dtmacro,binwidth,Texp)
macrolimits=macrotimesin
limits=macrolimits
taulist=np.zeros(len(limits)-1)
#wavelbins=150
#Exspeclist=np.zeros([wavelbins,len(limits)-1])
tauav=np.zeros(len(limits)-1)
Alist=np.zeros(len(limits)-1)
photonspbin=np.zeros(len(limits)-1)
ybglist=np.zeros(len(limits)-1)
buff=np.zeros(len(limits)-1)
meanex=np.zeros(len(limits)-1)
stdex=np.zeros(len(limits)-1)
histbinmultiplier=1
plt.figure()
lifetime1=rpl.GetLifetime(data[4],dtmicro,dtmacro,250e-9,tstart=-1,histbinmultiplier=1,ybg=0,plotbool=True,method='ML_c')
plt.show()
#[taulist,Alist,ybglist] = Parallel(n_jobs=-1, max_nbytes=None)(delayed(processInput)(tbinnr) for tbinnr in tqdm(range(nrtbins-1)))
#plt.figure()
#test=np.zeros((wavelbins,len(limits)-1))
for tbinnr in tqdm(range(len(limits)-1)):
    microtimes = microtimesin[limits[tbinnr]:limits[tbinnr+1]]
#    macrotimescycle = macrotimescyclein[limits[tbinnr]:limits[tbinnr+1]]
#    Exwavels=Wavelengthex=calibcoeffs[1]+InVoltage(macrotimescycle*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
#    meanex[tbinnr] = np.mean(Exwavels)
    #Plot ex spectrum for each time bin
#    histrange=(wavellimitlow,wavellimit)
#    wavelbins=300
#    [ylistex,xlistex] = np.histogram(Exwavels,wavelbins,histrange)
#    tlistex = (xlistex[:-1]+0.5*(xlistex[1]-xlistex[0]))
#    Exspeclist[:,tbinnr]=ylistex
#    plt.plot(tlistex,ylistex/max(ylistex)+tbinnr*0.1)
#    extime=macrotimescycleblue[limits[tbinnr]:limits[tbinnr+1]]
#    exwavel = calibcoeffs[1]+InVoltage(extime*data[1],Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
#    exrange = []
#    for pcounter in range(len(extime)-1):
#        if exwavel[pcounter]>bluelim and exwavel[pcounter]<redlim:
#            exrange.append(exwavel[pcounter])
#    meanex[tbinnr]=np.mean(exrange)
#    stdex[tbinnr]=np.std(exrange)
#    print(tbinnr)


    # if len(microtimes)>30:

    # [taulist[tbinnr],Alist[tbinnr],ybglist[tbinnr],buff[tbinnr]]=GetLifetime(microtimes,dtmicro,dtmacro,100e-9,tstart=lifetime[3]*dtmicro,histbinmultiplier=1,ybg=le(),plotbool=False,method='ML_c') 
    [taulist[tbinnr],Alist[tbinnr],ybglist[tbinnr],buff[tbinnr]]=rpl.GetLifetime(microtimes,dtmicro,dtmacro,300e-9,tstart=lifetime[3]*dtmicro,histbinmultiplier=1,ybg=lifetime[2]*binwidth/Texp,plotbool=False,method='ML_c') 

    # else:
    #     [taulist[tbinnr],Alist[tbinnr],ybglist[tbinnr],buff[tbinnr]]=GetLifetime(microtimes,dtmicro,dtmacro,60e-9,tstart=lifetime[3]*dtmicro,histbinmultiplier=1,ybg=0*lifetime[2]*binwidth/Texp,plotbool=False,method='ML_c') 

    tauav[tbinnr]=(np.mean(microtimes)-lifetime[3])*dtmicro*1e9-1
    photonspbin[tbinnr]=len(microtimes)
    #using th
    #for when histogramming photons
fig,ax1 = plt.subplots()
ax1.plot(macrotimesin[limits[0:len(limits)-1]]*dtmacro,photonspbin,'b')
#ax1.plot(macrotimesin[limits[0:len(limits)-1]]*dtmacro,Alist,'b')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Photons per '+str(int(binwidth*1000))+' ms bin', color='b')
#ax1.set_xlim([0,13])
#ax1.set_ylim([0,0.2*np.max(ylistspec)])
#ax1.ticklabel_format(style='sci',scilimits=(-2,3),axis='both')
ax1.tick_params('y', colors='b')
ax2 = ax1.twinx()
ax2.plot(macrotimesin[limits[0:len(limits)-1]]*dtmacro,tauav,'r')
#ax2.set_ylim([586,588])
ax2.set_ylim([0,50])
ax2.set_ylabel('Lifetime (ns)', color='r')
ax2.tick_params('y', colors='r')

#for when correlating emission    
fig,ax1 = plt.subplots()
ax1.plot(data[26][:-1]*dtmacro,photonspbin,'b')
#ax1.plot(macrotimesin[limits[0:len(limits)-1]]*dtmacro,Alist,'b')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Photons per emission bin', color='b')
#ax1.set_xlim([0,13])
#ax1.set_ylim([0,0.2*np.max(ylistspec)])
#ax1.ticklabel_format(style='sci',scilimits=(-2,3),axis='both')
ax1.tick_params('y', colors='b')
ax2 = ax1.twinx()
ax2.plot(data[26][:-1]*dtmacro,tauav,'r')
#ax2.set_ylim([586,588])
ax2.set_ylim([0,50])
ax2.set_ylabel('Lifetime (ns)', color='r')
ax2.tick_params('y', colors='r')
#ax2.ticklabel_format(style='sci',scilimits=(-2,3),axis='both')


#%% Plot FLID map
plt.figure()
plt.hist2d(taulist,photonspbin,(50,int(np.max(photonspbin))),range=[[0,80],[0,ceil(np.max(photonspbin))]],norm=mcolors.LogNorm())
plt.title('FLID map')
plt.ylabel('Counts per bin')
plt.xlabel('Lifetime (ns)')
plt.colorbar()

#%% Plot histogram
plt.figure()
histdata=rpl.Easyhist(photonspbin,0,np.max(photonspbin),2)
plt.plot(histdata[0],histdata[1])
plt.xlabel('Photons per '+str(int(binwidth*1000))+' ms bin')
plt.ylabel('Occurence')
#%% Limits
limitex=500
limittrlow=42
limittrhigh=80
limitoffhigh=20
# microtimesin=microtimesblue
# macrotimesin=macrotimesblue
macrotimescyclein=data[19]
microtimes_ex= np.zeros(len(data[2]),dtype='int64')
macrotimescycle_ex= np.zeros(len(data[2]),dtype='int64')
nrex=0
bins_ex=0
microtimes_trion= np.zeros(len(data[2]),dtype='int64')
macrotimescycle_trion= np.zeros(len(data[2]),dtype='int64')
nrtrion=0
bins_mid=0
microtimes_mid= np.zeros(len(data[2]),dtype='int64')
macrotimescycle_mid= np.zeros(len(data[2]),dtype='int64')
nrmid=0
bins_off=0
microtimes_off= np.zeros(len(data[2]),dtype='int64')
macrotimescycle_off= np.zeros(len(data[2]),dtype='int64')
nroff=0
bins_trion=0
nrtrlate=0
bins_trlate=0
microtimes_trlate= np.zeros(len(data[2]),dtype='int64')
macrotimescycle_trlate= np.zeros(len(data[2]),dtype='int64')
nrtrearly=0
bins_trearly=0
microtimes_trearly= np.zeros(len(data[2]),dtype='int64')
macrotimescycle_trearly= np.zeros(len(data[2]),dtype='int64')
timetrion = np.zeros(len(limits)-1)
timeexciton = np.zeros(len(limits)-1)
for tbinnr in range(len(limits)-1):
    nphots = limits[tbinnr+1]-limits[tbinnr]
    microtimes = microtimesin[limits[tbinnr]:limits[tbinnr+1]]
    macrotimescycle = macrotimescyclein[limits[tbinnr]:limits[tbinnr+1]]

    if nphots>limitex and taulist[tbinnr]<50:

        microtimes_ex[nrex:nrex+nphots]=microtimes
        macrotimescycle_ex[nrex:nrex+nphots]=macrotimescycle
        bins_ex+=1
        nrex+=nphots
        timeexciton[tbinnr]=1


    elif nphots>limittrlow and nphots<limittrhigh and taulist[tbinnr]<10 and tauav[tbinnr]>1:# and taulist[tbinnr]>0.05: #and (photonspbin[tbinnr]-7)/taulist[tbinnr]>112/28

#    elif photonspbin[tbinnr]>limittrlow and photonspbin[tbinnr]<limittrhigh and taulist[tbinnr]>0.5 and (photonspbin[tbinnr]-10)/taulist[tbinnr]>505/27:
        microtimes_trion[nrtrion:nrtrion+nphots]=microtimes
        macrotimescycle_trion[nrtrion:nrtrion+nphots]=macrotimescycle
        bins_trion+=1
        nrtrion+=nphots
        timetrion[tbinnr]=1
        for k in range(len(microtimes)):
            if microtimes[k]*dtmicro>22e-9 and microtimes[k]*dtmicro<50e-9:
                # print('found')
                microtimes_trlate[nrtrlate]=microtimes[k]
                macrotimescycle_trlate[nrtrlate]=macrotimescycle[k]
                nrtrlate+=1
            if microtimes[k]*dtmicro>9e-9 and microtimes[k]*dtmicro<15e-9:
                # print('found')
                microtimes_trearly[nrtrearly]=microtimes[k]
                macrotimescycle_trearly[nrtrearly]=macrotimescycle[k]
                nrtrearly+=1
    # elif photonspbin[tbinnr]>limittrlow and photonspbin[tbinnr]<limitex and tauav[tbinnr]>25 :
    elif nphots<limitoffhigh and taulist[tbinnr]<10:
        microtimes_off[nroff:nroff+nphots]=microtimes
        macrotimescycle_off[nroff:nroff+nphots]=macrotimescycle
        bins_off+=1
        nroff+=nphots
    if nphots>limitex and taulist[tbinnr]>60:
        # print('found')
        microtimes_mid[nrmid:nrmid+nphots]=microtimes
        macrotimescycle_mid[nrmid:nrmid+nphots]=macrotimescycle
        bins_mid+=1
        nrmid+=nphots
microtimes_ex = microtimes_ex[:nrex]
macrotimescycle_ex = macrotimescycle_ex[:nrex]
microtimes_trion = microtimes_trion[:nrtrion]
macrotimescycle_trion = macrotimescycle_trion[:nrtrion]
microtimes_mid = microtimes_mid[:nrmid]
macrotimescycle_mid = macrotimescycle_mid[:nrmid]
microtimes_off = microtimes_off[:nroff]
macrotimescycle_off = macrotimescycle_off[:nroff]
microtimes_trlate = microtimes_trlate[:nrtrlate]
macrotimescycle_trlate = macrotimescycle_trlate[:nrtrlate]
microtimes_trearly = microtimes_trearly[:nrtrearly]
macrotimescycle_trearly = macrotimescycle_trearly[:nrtrearly]
#% Lifetime of Exciton and Trion
plt.figure()
fitex=rpl.GetLifetime(microtimes_ex,dtmicro,dtmacro,200e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_ex/(Texp/binwidth),method='ML_c')
# fitex=GetLifetime(microtimes_ex,dtmicro,dtmacro,100e-9,tstart=-1,plotbool=True,ybg=bins_ex*40*binwidth/np.max(microtimesblue),method='ML_c')
fittrion=rpl.GetLifetime(microtimes_trion,dtmicro,dtmacro,5e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_trion/(Texp/binwidth),method='ML_c')
# fittrion=GetLifetime(microtimes_trion,dtmicro,dtmacro,5e-9,tstart=-1,plotbool=True,ybg=bins_trion*40*binwidth/np.max(microtimesblue),method='ML_c')

# fitoff=rpl.GetLifetime(microtimes_off,dtmicro,dtmacro,10e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_trion/(Texp/binwidth),method='ML_c')
fitmid=rpl.GetLifetime(microtimes_off,dtmicro,dtmacro,200e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_mid/(Texp/binwidth),method='ML_c')

print('Rad lifetime ratio:'+str(fittrion[1]/bins_trion/(fitex[1]/bins_ex)))
#plt.xlim([0,220])
# plt.legend(['High cps','High cps fit','Mid cps','Mid cps fit','Low cps','Low cps fit'])
#%% emission spectrum when in trion state
indextrion = np.argwhere(timetrion==1).ravel()
indexexciton = np.argwhere(timeexciton==1).ravel()
sumtrion=np.mean(Yfiltered[:,indextrion],axis=1)
plt.figure()
plt.plot(wavelengths,sumtrion)
plt.title('trion PLE')

inttemp=0
wavtemp=0
for j in range(len(wavelengths)):
    inttemp += wavelengths[j]*sumtrion[j]
    wavtemp += sumtrion[j]
centertrion=inttemp/wavtemp
print(centertrion)

sumexciton =np.mean(Yfiltered[:,indexexciton],axis=1)
plt.figure()
plt.plot(wavelengths,sumexciton)
plt.title('exciton PLE')

inttemp=0
wavtemp=0
for j in range(len(wavelengths)):
    inttemp += wavelengths[j]*sumexciton[j]
    wavtemp += sumexciton[j]
centerexciton=inttemp/wavtemp
print(centerexciton)
#%%
taus=[0,1,2]
# plot='corr'
plot='none'
Emissiondata1=Yfiltered[:,indextrion]
# Emissiondata2=Yfiltered[:,indextrion]

# Emissiondata1=Yfiltered[:,indexexciton]
# Emissiondata2=Yfiltered[:,indexexciton]
Emissiondata2=excitationdata[0][:,indextrion]
emissioncovariance,emissionnormalization,emissioncorrelation = rplm.pearsoncorrelation(Emissiondata1,Emissiondata2,wavelengths,wavelengths,taus=taus,plot=plot)

# for j in range(taus):
for i in range(len(taus)):
    tau=taus[i]
    # if tau==0:
    # vmax0corr=np.nanmax(np.delete(emissioncorrelation.ravel(),np.where(emissioncorrelation.ravel()>=0.95))) #done to remove the whole =1 diagonal visualization and allows for better visualization of the plot
    vmin=-0.2
    vmax0corr=0.4
    # print(vmax0corr)
    plt.figure()
    plt.imshow(emissioncorrelation[i],extent=[np.min(excitationdata[1][excitationdata[1]>2]),np.max(excitationdata[1]),np.max(wavelengths),np.min(wavelengths)],vmin=vmin,vmax=vmax0corr)
    # plt.imshow(emissioncorrelation[i],extent=[np.min(wavelengths),np.max(wavelengths),np.max(wavelengths),np.min(wavelengths)],vmin=vmin,vmax=vmax0corr)
    # plt.imshow(emissionnormalization[i],extent=[np.min(wavelengths),np.max(wavelengths),np.max(wavelengths),np.min(wavelengths)])
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Wavelength (nm)')
    plt.title('Correlation map. tau = '+str(taus[i]))
    
    #%% find origin on data to see whether trion also diffuses
guessorigin=620
origin,rest=rplm.find_origin(emissionnormalization[1],guessorigin,wavelengths)
print(wavelengths[origin])
    
#%%
histsettings=(420,600,0.5)
plt.figure()
exspec=rpl.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_ex*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
plt.plot(exspec[0],exspec[1]/np.sum(exspec[1]))
exspec=rpl.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_trion*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
plt.plot(exspec[0],exspec[1]/np.sum(exspec[1]))
# exspec=rpl.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_mid*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
# plt.plot(exspec[0],(exspec[1]/np.sum(exspec[1])))
#%% loglog decay
t0=lifetime[3]*dtmicro*1e9
timetrace=rpl.Easyhist(microtimes_trion*dtmicro,0,dtmicro*1000,dtmicro)
plt.figure()
plt.semilogy((timetrace[0])*1e9-t0,timetrace[1])
# plt.loglog(tvalues,lifetime[1][0]*np.exp(-tvalues/lifetime[0][0])/20)
# plt.loglog(tvalues,lifetime[1][1]*np.exp(-tvalues/lifetime[0][1])/40)
# plt.plot(tvalues+lifetime[3]*dtmicro*1e9-t0,lifetime[1][2]*np.exp(-tvalues/lifetime[0][2]))
# plt.semilogy((timetrace[0])*1e9,2e1*np.exp(-(timetrace[0])*1e9/(115)))
# plt.ylim(1e0,1e3)
#%% microtime-gated PLE
histsettings=(440,612,1)
microtimelims=[0.1,5]
rangeindices=np.argwhere(((microtimes_trion*dtmicro*1e9-t0)>microtimelims[0]) &((microtimes_trion*dtmicro*1e9)-t0<microtimelims[1]))
exspecA=rpl.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_trion[rangeindices]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
plt.figure()
plt.plot(exspecA[0],(exspecA[1]-np.mean(exspecA[1][160:170]))/np.sum(exspecA[1][:55]-np.mean(exspecA[1][160:170])))#/interprefspec(exspecA[0]))


microtimelims=[15,50]
exspecB=rpl.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_ex*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
# rangeindices=np.argwhere(((microtimes0*dtmicro*1e9-t0)>microtimelims[0]) &((microtimes0*dtmicro*1e9)-t0<microtimelims[1]))
# exspecB=rpl.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(data[19][rangeindices]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
plt.plot(exspecB[0],(exspecB[1]-np.mean(exspecB[1][160:170]))/np.sum(exspecB[1][:55]-np.mean(exspecB[1][160:170])))#/interprefspec(exspecB[0]))

#%%
testA=(exspecA[1]-np.mean(exspecA[1][160:170]))/np.sum(exspecA[1][:55]-np.mean(exspecA[1][160:170]))
testB=(exspecB[1]-np.mean(exspecB[1][160:170]))/np.sum(exspecB[1][:55]-np.mean(exspecB[1][160:170]))
plt.plot(exspecB[0],(testA/interprefspec(exspecA[0])-testB/interprefspec(exspecB[0]))/(testB/interprefspec(exspecB[0])))
plt.plot(exspecB[0],np.zeros(len(exspecB[0])),'--k')
# plt.plot(exspecB[0],(exspecB[1]-np.min(exspecB[1]))/np.sum(exspecB[1][20:100]-np.min(exspecB[1]))-(exspecA[1]-np.min(exspecA[1]))/np.sum(exspecA[1][20:100]-np.min(exspecA[1])))
#%% Wavelength filtered data
histsettings=(440,600,3)
maxwavel=580
indices0=np.argwhere((calibcoeffs[1]+rpl.InVoltagenew(data[19]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0])<580)
exspec=rpl.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(data[19][indices0]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
# plt.plot(exspec[0],(exspec[1]-np.min(exspec[1]))/np.sum(exspec[1]-np.min(exspec[1])))
indices1=np.argwhere((calibcoeffs[1]+rpl.InVoltagenew(data[20]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0])<580)
g2data=rpl.MakeG2(times0[indices0],times1[indices1],dtmicro,g2restime=8*dtmicro,nrbins=1000)
plt.plot(g2data[0],g2data[1])
