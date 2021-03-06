# -*- coding: utf-8 -*-

"""
Created on Thu Mar 28 16:44:49 2019

@author: rober
"""
import os, numpy as np, csv, matplotlib.pyplot as plt, scipy.optimize as opt, math, struct, binascii, gc, time, random
import numpy.matlib
import sys
import multiprocessing
from operator import sub
from joblib import Parallel, delayed
import scipy#, lmfit
from scipy import stats
from scipy.optimize import minimize, curve_fit # used for implementation of maximum likelihood exponential fit
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
from mpl_toolkits import axes_grid1
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.signal import find_peaks, peak_widths
from scipy import fftpack
import re
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from lmfit.models import LorentzianModel, GaussianModel, VoigtModel, LinearModel, ConstantModel
from lmfit.models import PolynomialModel
from lmfit import Model

import socket #enables to find computer name to make less of a mess with folders
#%matplotlib auto
sys.path.append('E:/Martijn/ETH/scripts/github/martijn_own_github/repoo/apd_ple')
import Read_PLE_functions as rpl
sys.path.append('E:/Martijn/ETH/scripts/github/martijn_own_github/repoo/martijn_own_github')
import Read_PLE_functions_Martijn as rplm
InVoltagenew_c=nb.jit(nopython=True)(rplm.InVoltagenew)



# In[25]:


### GENERAL EVALUATION OF TTTR data     

# parameters (use forward slashes!)
Debugmode=False     #Set to true to make plots in the individual sections of the code to check that everything is working properly

basefolders={'DESKTOP-BK4HAII':'C:/Users/rober/Documents/Doktorat/Projects/SingleParticle_PLE/Andor Spectrometer/',
             'HP_Probook':'E:/Martijn/ETH/results/',
             'mavt-omel-w004w':'E:/LAB_DATA/Robert/'} #dictionary which base folder to use on which computer. For you this probably would be everything until 
folder = basefolders[socket.gethostname()]+'20200630_CdSe_cryo_HR2/'

filename ='HR2_QD7_9p734MHz_200Hz_260mVpp_n20mVoff_ND0'

settingsfile= filename+'_settings'
HHsettings=rpl.load_obj(settingsfile, folder )
# HHsettings={}

# Texp = 240 # total time [s] of measurements
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
# data = rpl.ImportT3(folder + filename + '.ptu',HHsettings)

Texp = round(data[8]*data[1]*1024,0) # macrotime values can store up to 2^10 = 1024 before overflow. overflows*1024*dtmacro gives experiment time [s]
print('averaged cps on det0 and det1:',np.array(data[6:8])/Texp)
print('experimental time in s:',Texp)
[microtimes0,microtimes1,times0,times1,dtmicro,dtmacro,decaytlist,decayylist] = rpl.ShiftPulsedData(data[2],data[4],data[3],data[5],data[0],data[1]) #decaytlist and decayylist are the two variables you want to check for the modulation trace
# MaxLikelihoodFunction_c = nb.jit(nopython=True)(rpl.MaxLikelihoodFunction)
plt.figure()
lifetime=rpl.GetLifetime(microtimes0,dtmicro,dtmacro,dtfit=400e-9,tstart=-1,histbinmultiplier=1,ybg=-1,plotbool=True,expterms=2,method='ML_c')

limits0 = rpl.HistPhotons(times0*dtmicro,binwidth,Texp) #gives index of first photon of det0 in each bin
limits1 = rpl.HistPhotons(times1*dtmicro,binwidth,Texp)

# make an intensity trace and find Imax
inttrace = rpl.MakeIntTrace(limits0,limits1,binwidth,Texp)


#%% Wavelength calibration

#%
Vpp=float(filename.split('mVpp_')[0].split('_')[-1])
Freq=float(filename.split('Hz_')[-2])
# Voffset=80
if filename.split('mVoff')[0].split('Vpp_')[-1].startswith('n'):
    Voffset=float(filename.split('mVoff')[0].split('Vpp_')[-1].split('n')[-1])*-1
else:
    Voffset=float(filename.split('mVoff')[0].split('Vpp_')[-1])
# Voffset=0
#this dataset and correctionw as in folder E:/Martijn/ETH/results/20200630_CdSe_cryo_HR2/'
if folder=='E:/Martijn/ETH/results/20200630_CdSe_cryo_HR2':
    #this folder was chosen because it had the most complete dataset
    files=['backref_9p734MHz_0Hz_0mVpp_20mVoff_ND0.asc','backref_9p734MHz_0Hz_0mVpp_60mVoff_ND0.asc','backref_9p734MHz_0Hz_0mVpp_100mVoff_ND0.asc','backref_9p734MHz_0Hz_0mVpp_140mVoff_ND0.asc','backref_9p734MHz_0Hz_0mVpp_180mVoff_ND0.asc','backref_9p734MHz_0Hz_0mVpp_n20mVoff_ND0.asc','backref_9p734MHz_0Hz_0mVpp_n60mVoff_ND0.asc','backref_9p734MHz_0Hz_0mVpp_n100mVoff_ND0.asc']
    Voltage=np.zeros(len(files))
    Wavelength_calib=np.zeros(len(files))
    
    for idx in range(len(files)):
        file=files[idx]
        backref1=rpl.importASC(folder+file)
        Wavelength_calib[idx]=float(backref1[1][np.unravel_index(backref1[0].argmax(),backref1[0].shape)[1]])
        if file.split('mVoff')[0].split('Vpp_')[-1].startswith('n'):
            Voffset=float(file.split('mVoff')[0].split('Vpp_')[-1].split('n')[-1])*-1
        else:
            Voffset=float(file.split('mVoff')[0].split('Vpp_')[-1])
        Voltage[idx]=Voffset
    # print(Wavelength_calib[idx],Voltage[idx])
else:        
    Voltage=np.asarray([  20.,   60.,  100.,  140.,  180.,  -20.,  -60., -100.])
    Wavelength_calib=np.asarray([532.91718, 514.74475, 496.27448, 477.65393, 458.96927, 551.38086,
           569.95935, 588.56512])
    print('Warning. The calibration was done in a different folder')
calibcoeffs=np.polyfit(Voltage,Wavelength_calib,1)
slope, intercept, r_value, p_value, std_err = stats.linregress(Voltage,Wavelength_calib)

if Debugmode==True:
    plt.figure()
    plt.scatter(Voltage,Wavelength_calib)
    plt.xlabel('Voltage (mV)')
    plt.ylabel('Excitation wavelength (nm)')
    plt.plot(Voltage,calibcoeffs[1]+Voltage*calibcoeffs[0],Voltage,Wavelength_calib,'*')
    plt.legend(['Linear Fit','Measurement'])
    plt.title('Voltage to wavelength')
    plt.figtext(7/10,7/10,'R_sq = '+str(r_value**2)[:6])
    
# matchrange=(440,505)    #perovskites
matchrange=(500,600)    #QDs. Real range does not matter so much
tshift=rplm.Findtshift(Freq,Vpp,Voffset,calibcoeffs,data[19],dtmacro,matchrange,(-6e-4,-2e-4)) #coarse sweep
tshift=rplm.Findtshift(Freq,Vpp,Voffset,calibcoeffs,data[19],dtmacro,matchrange,(tshift-1e-5,tshift+1e-5),Debugmode=True)


print('Voffset =',Voffset,'filename=',filename)    #just to check whether the Voffset was real
#%% laser intensity corrected
#combines exctiation wavelength calibration on Andor spectrometer
    #done by sweeping laser spectrum over camera and divide QD em spectrum over normalized reference spectrum
reffile = str(filename.split('MHz')[0].split('_')[-1])
reffolder = folder#basefolders[socket.gethostname()]+'20200622_Perovskites_cryo/calibrationArio$/'
# calibspec = rpl.importASC(reffolder+'backref_77p87MHz_200Hz_200mVpp_145mVoff_SP_no450LP.asc')
# calibspec = rpl.importASC(reffolder+'backref_4p86MHz_200Hz_500mVpp_60mVoffset.asc')
calibspec = rpl.importASC(reffolder+'backref_9p734MHz_200Hz_500mVpp_60mVoff_ND0.asc')


cmaxpos=np.argmax(np.mean(calibspec[0][:,:],1))
refspecold=np.mean(calibspec[0][cmaxpos-5:cmaxpos+5,:],0)
refspec=np.mean(calibspec[0][cmaxpos-5:cmaxpos+5,:],0)#-np.mean(refspecold[480:len(refspecold)])
refspec=refspec*1.2398/calibspec[1]
refspec=refspec/np.max(refspec)
# refspec=savgol_filter(refspec, 23, 3)
interprefspec= interp1d(calibspec[1], refspec,kind='cubic',fill_value='extrapolate')
plt.figure()
# # plt.plot(np.linspace(500,600,300),interprefspec(np.linspace(500,600,300)))
plt.plot(calibspec[1],refspec)
plt.title('Spectrum of laser')
plt.xlabel('Excitation wavelength (nm)')
plt.ylabel('Intensity')

wavelengthrangemin,wavelengthrangemax = 430,595

wavelengthrange = (wavelengthrangemin,wavelengthrangemax)
histogrambin=247
# wavelengthrange = matchrange
[excspec,excspecwavtemp] = np.histogram(InVoltagenew_c(dtmacro*data[19],Freq,Vpp,Voffset,tshift)*calibcoeffs[0]+calibcoeffs[1],histogrambin,range=wavelengthrange)
excspecwav = [excspecwavtemp[:-1]+(excspecwavtemp[1]-excspecwavtemp[0])/2]
excspeccorr = excspec/interprefspec(excspecwav)

plt.figure()
plt.plot(excspecwav[0],excspec/np.max(excspec))    
plt.title('Uncorrected') 
plt.xlabel('Excitation wavelength (nm)')
plt.ylabel('Intensity')
plt.figure()
plt.plot(excspecwav[0],excspeccorr[0]/np.max(excspeccorr[0]))   
plt.title('Laser corrected') 
plt.xlabel('Excitation wavelength (nm)')
plt.ylabel('Intensity')
    

#%% Read emission spectra
    
emissionspec = rpl.importASC(folder+filename+'.asc')
# emissionspec = rpl.importASC(folder+'Kineticseries_LED_Fluocube_midPower_noND.asc')
#produces array of emission spectra 
wavelengthsinit = emissionspec[1]
numberypixels = len(emissionspec[0])
numberspectra = len(emissionspec[0][0][0])


#%% producing Yfiltered and reorganizing spectra
if len(emissionspec[0])==1:
    # Ybacktemp = np.argmin(np.sum(emissionspec[0][0],axis=1))
    # Ybackground=np.min(np.mean(emissionspec[0],axis=2))
    # Ybackground=500
    wavelengths = wavelengthsinit
    timeaverage= np.mean(emissionspec[0],axis=2)
    Ybackground=rplm.fittimeaverage(timeaverage.ravel(),wavelengths,'Lor') #done to find the background
    plt.title('Fit used for background estimation')
    Yfiltered = emissionspec[0][0]-Ybackground
    # wavelengths = wavelengthsinit
    numberypixels=1
    # timeaverage = np.mean(emissionspec[0],axis=2) -Ybackground
    
    print('Estimated background : ', Ybackground)
    
    if Debugmode==True:
        plt.figure()
        plt.plot(wavelengths,timeaverage.ravel()/np.max(timeaverage))
        plt.xlabel('Emission wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title('Emission spectrum')
        
        emwavelrange=(595,620)
        emwavelindices=(np.argmin(np.abs(wavelengths-emwavelrange[0])),np.argmin(np.abs(wavelengths-emwavelrange[1])))
        timeminplot=0
        timemaxplot=-1
        plt.figure()
        plt.imshow(Yfiltered[emwavelindices[0]:emwavelindices[1],timeminplot:timemaxplot],extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths[emwavelindices[0]:emwavelindices[1]]),np.min(wavelengths[emwavelindices[0]:emwavelindices[1]])],aspect='auto')#,vmax=1000)
        plt.scatter(np.linspace(data[26][0]*dtmacro,data[26][-1]*dtmacro,len(Yfiltered[0])),fits[2], c='r',s=1)
        # plt.imshow(Yfiltered,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.min(wavelengths),np.max(wavelengths)],aspect='auto')
        # plt.imshow(Yfiltered)
        plt.ylabel('Emission wavelength (nm)')
        plt.xlabel('Time (s)')
        plt.title('Emission spectrum vs time')
        plt.gca().invert_yaxis()
        plt.colorbar()
    
else:
    avg = np.mean(emissionspec[0],axis=2) 

    if Debugmode==True:
        plt.figure()
        im = plt.imshow(avg,extent=[wavelengthsinit[0],wavelengthsinit[-1],0,numberypixels])
        plt.xlabel('Emission wavelength (nm)')
        plt.ylabel('Y-pixel')
        plt.gca().invert_yaxis()
        rplm.add_colorbar(im) #color image showing the ypixels of interest, to increase the signal to noise ratio
    
    emissionlines = np.argwhere(avg==np.max(avg))
    Ypixelmax = emissionlines[0][0] #automatized value finder for the ypixels
    Wavelengthmax = emissionlines[0][1] 
    removedinit = Ypixelmax-5 
    startfinal = Ypixelmax+5 #maybe change this. But these are the ypixels of interest, based on above color image. If you want something more rigorous it should include an if statement where the background is approximately equal to the 'Questioned Background'. In my opinion it is quite unecesary
    Yfilteredinit = np.mean(emissionspec[0][removedinit:startfinal],axis=0)
    timeaverageinit = np.mean(Yfilteredinit,axis=1)
    
    meanYpixels = np.zeros(numberypixels)
    for j in range(numberypixels):
        meanYpixels[j] = np.mean(emissionspec[0][j])
        
    bkgbuffer = 5 # number for allowing backgrounds    
    ybackgroundindex = np.argwhere(meanYpixels<np.min(meanYpixels)+bkgbuffer)[:,0]
    Ybackgroundinit = np.zeros((len(ybackgroundindex),len(wavelengthsinit)))
    for j in range(len(ybackgroundindex)):
        Ybackgroundinit[j] = np.mean(emissionspec[0][ybackgroundindex[j]],axis=1)
        
    
    Ybackground = np.mean(Ybackgroundinit) #automatizes finding background based on minimum+buffer
    print('Estimated background : ', Ybackground)
    # Ypixelbackgroundinit = 
        
    # Ypixelbackgroundinit = removedinit-5 #this only works when there is one emitter in the image
    # Ypixelbackgroundfinal = startfinal+5 #additional measure of background
    
    # # Ypixelbackgroundinit = 50
    # # Ypixelbackgroundfinal = 150
    # Ybackground = np.mean([np.mean(np.mean(emissionspec[0][0:Ypixelbackgroundinit],axis=0)),np.mean(np.mean(emissionspec[0][Ypixelbackgroundfinal:-1],axis=0))]) #calculates one background from the entire series, neglecting dependency of the pixel or the time of the experiment
    # # timeaverageinit = np.mean(Yfilteredinit,axis=1)
    
    nrspec = 0#first so many emission wavelengths that are neglected
    nrfinspec = -1 #last so many emission wavelengths that are neglected
    
    wavelengths = wavelengthsinit[nrspec:nrfinspec]
    Yfilteredwithbkg = Yfilteredinit[nrspec:nrfinspec]
    timeaveragewithbkg = timeaverageinit[nrspec:nrfinspec]
    
    Yfiltered = Yfilteredwithbkg-Ybackground #Yfilterd is now background corrrected. May therefore have some negative values
    # Yfiltered[Yfiltered<=0]=0 #I think removing all the negative values should help the readability of the map
    timeaverage = timeaveragewithbkg-Ybackground
    
    if Debugmode==True:
        plt.figure()
        plt.plot(wavelengths,timeaverage)
        plt.xlabel('Emission wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title('time-averaged signal')
        
        plt.figure()
        plt.imshow(Yfiltered,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths),np.min(wavelengths)])
        # plt.imshow(Yfiltered)
        plt.ylabel('Emission wavelength (nm)')
        plt.xlabel('Time (s)')
        plt.title('intensity vs time per wavelength')
        plt.gca().invert_yaxis()
        
    
        
    if Debugmode==True:
        plt.figure()
        plt.plot(np.linspace(data[26][0]*dtmacro,data[26][-1]*dtmacro,len(Yfiltered[0])),np.mean(Yfiltered,axis=0))
        plt.xlabel('Time (s)')
        plt.ylabel('Intensity')
        plt.title('Intensity trace')    
        
    if Debugmode==True:
        plt.figure()
        plt.xlabel('Emission wavelength (nm)')
        plt.ylabel('Intensity')
        for i in range(100,200,1): #select some spectra you want to see
            plt.plot(wavelengths,Yfiltered[:,i])

#correcting for trigger events longer than captured spectra
if Yfiltered.shape[1]==data[26].shape[0]:
    pass
elif Yfiltered.shape[1]<=data[26].shape[0]:
    temp=np.zeros((len(Yfiltered),len(data[26])))
    a=int(np.abs(len(data[26])-len(Yfiltered[0])))
    temp[:,a:]=Yfiltered
    Yfiltered=temp
    
elif Yfiltered.shape[1]>=data[26].shape[0]:
    # temp=np.zeros((len(Yfiltered),len(data[26])))
    a=len(data[26])
    # temp[:,:a]=Yfiltered
    Yfiltered=Yfiltered[:,:a]
    
    #Yfiltered is in bins of (wavelength/vertical,time/horizontal)
#%%some fits
fits=rplm.fitspectra(Yfiltered,wavelengths,0,len(Yfiltered[0]),'Lor',Debugmode=False)

#%% Generate binned excitation spectra for excitation emission correlation
# #select particular exctiation wavelength based on chosen range emission wavelengths


rawdata = data[25]
originaldata= data[19]
wavelengthrange = matchrange
# wavelengthrange = (430,575)
wavelengthrange = (500,575)
histogram = 300
excitationdata1 = rplm.histogramexcitationspectra(Freq,Vpp,Voffset,tshift,calibcoeffs,dtmacro,rawdata,originaldata,wavelengthrange,histogram)
originaldata= data[20]
excitationdata2 = rplm.histogramexcitationspectra(Freq,Vpp,Voffset,tshift,calibcoeffs,dtmacro,rawdata,originaldata,wavelengthrange,histogram)
# excitationdata = histogramexcitationspectranorm(rawdata,originaldata,wavelengthrange,histogram,interprefspec)
excitationdata=excitationdata1[0]+excitationdata2[0]
# excitationdata=excitationdata2[0]
excitationdatacorr=np.zeros(excitationdata.shape)
for j in range(len(excitationdata[0])):
    excitationdatacorr[:,j]=excitationdata[:,j]/interprefspec(excitationdata1[1][:,0])
# excitationdata2=np.zeros(excitationdata.shape)
# for j in range(len(excitationdata[0])):
#     excitationdata2[:,j] = excitationdata[:,j]-np.mean(excitationdata[:,j])

# maxexcitation = np.max(excitationdata[0])    
vmax=np.max(excitationdata)
vmax=60
plt.figure()
plt.imshow(excitationdata,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,wavelengthrange[1],wavelengthrange[0]],vmin=0,vmax=vmax,aspect='auto')
plt.gca().invert_yaxis()
plt.colorbar()
plt.ylabel('Excitation Wavelength (nm)')
plt.xlabel('time (s)')
plt.title('Full excitation spectra')  

#%% emission and excitaiton together
#each plotted in time frame

    
# if Debugmode==True:
fig,ax=plt.subplots(2,1,sharex=True,sharey=False,constrained_layout=True) #plot excitation and emission above each other. If you want them to be visualized with the same window the yrange should cover the same distance. also zooming goes automatically
# emwavelrange=(500,540)
emwavelrange=(590,620)
emwavelindices=(np.argmin(np.abs(wavelengths-emwavelrange[0])),np.argmin(np.abs(wavelengths-emwavelrange[1])))
timeminplot=0
timemaxplot=-1
vmax=2000
vmax=np.max(Yfiltered[Yfiltered<np.max(Yfiltered)])
vmax=np.max(Yfiltered)
# plt.figure()
# plt.imshow(Yfiltered[emwavelindices[0]:emwavelindices[1],timeminplot:timemaxplot],extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths[emwavelindices[0]:emwavelindices[1]]),np.min(wavelengths[emwavelindices[0]:emwavelindices[1]])],aspect='auto')#,vmax=1000)
# im0=ax[0].imshow(Yfiltered,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths),np.min(wavelengths)],aspect='auto')
im0=ax[0].imshow(Yfiltered[emwavelindices[0]:emwavelindices[1],:],extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths[emwavelindices[0]:emwavelindices[1]]),np.min(wavelengths[emwavelindices[0]:emwavelindices[1]])],aspect='auto',vmin=0)
# im0=ax[0].imshow(Yfiltered,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths),np.min(wavelengths)])
ax[0].set_ylabel('Emission wavelength (nm)')
ax[0].set_xlabel('Time (s)')
ax[0].set_title('Emission')
ax[0].xaxis.set_tick_params(which='both', labelbottom=True)
ax[0].invert_yaxis()
fig.colorbar(im0,ax=ax[0])
# vmax=50
vmax=np.max(excitationdatacorr)
vmax=np.max(excitationdata)
im1=ax[1].imshow(excitationdata,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(excitationdata1[1]),np.min(excitationdata1[1][excitationdata1[1]>0])],aspect='auto',vmin=0,vmax=vmax)
# im1=ax[1].imshow(excitationdata[0],extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(excitationdata[1]),np.min(excitationdata[1][excitationdata[1]>0])])
ax[1].set_ylabel('Excitation wavelength (nm)')
ax[1].set_xlabel('Time (s)')
ax[1].set_title('Excitation')
fig.colorbar(im1,ax=ax[1])
ax[1].invert_yaxis()
# plt.tight_layout()
    
    
fig,ax=plt.subplots(2,1,sharex=True,sharey=False,constrained_layout=True) #plot excitation and emission above each other. If you want them to be visualized with the same window the yrange should cover the same distance. also zooming goes automatically
# emwavelrange=(600,640)
# emwavelindices=(np.argmin(np.abs(wavelengths-emwavelrange[0])),np.argmin(np.abs(wavelengths-emwavelrange[1])))
timeminplot=0
timemaxplot=-1
vmax=2000
vmax=np.max(Yfiltered)
# plt.figure()
# plt.imshow(Yfiltered[emwavelindices[0]:emwavelindices[1],timeminplot:timemaxplot],extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths[emwavelindices[0]:emwavelindices[1]]),np.min(wavelengths[emwavelindices[0]:emwavelindices[1]])],aspect='auto')#,vmax=1000)
# im0=ax[0].imshow(Yfiltered,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths),np.min(wavelengths)],aspect='auto')
im0=ax[0].imshow(Yfiltered[emwavelindices[0]:emwavelindices[1],:],extent=[0,len(Yfiltered[0]),np.max(wavelengths[emwavelindices[0]:emwavelindices[1]]),np.min(wavelengths[emwavelindices[0]:emwavelindices[1]])],aspect='auto')
# im0=ax[0].imshow(Yfiltered,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths),np.min(wavelengths)])
ax[0].set_ylabel('Emission wavelength (nm)')
ax[0].set_xlabel('Index')
ax[0].set_title('Emission')
ax[0].xaxis.set_tick_params(which='both', labelbottom=True)
ax[0].invert_yaxis()
fig.colorbar(im0,ax=ax[0])
vmax=np.max(excitationdatacorr)
vmax=np.max(excitationdata)
im1=ax[1].imshow(excitationdatacorr,extent=[0,len(excitationdata[0]),np.max(excitationdata1[1]),np.min(excitationdata1[1][excitationdata1[1]>0])],vmin=0,vmax=vmax,aspect='auto')
# im1=ax[1].imshow(excitationdata[0],extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(excitationdata[1]),np.min(excitationdata[1][excitationdata[1]>0])])
ax[1].set_ylabel('Excitation wavelength (nm)')
ax[1].set_xlabel('Index')
ax[1].set_title('Excitation')
fig.colorbar(im1,ax=ax[1])
ax[1].invert_yaxis()
   

#select excitation spectra based on trigger events from first photons in the emission cycle
#can do excitation emission correlation only on those. Forward and backward sweep combined
#%% emission and excitaiton together for iamges
#each plotted in time frame

# finalindex=7000    
# if Debugmode==True:
fig,ax=plt.subplots(2,1,sharex=True,sharey=False,constrained_layout=True) #plot excitation and emission above each other. If you want them to be visualized with the same window the yrange should cover the same distance. also zooming goes automatically
# emwavelrange=(515,540)
emwavelrange=(590,620)
emwavelindices=(np.argmin(np.abs(wavelengths-emwavelrange[0])),np.argmin(np.abs(wavelengths-emwavelrange[1])))
timeminplot=0
timemaxplot=-1
vmax=2000
vmax=np.max(Yfiltered[Yfiltered<np.max(Yfiltered)])
vmax=np.max(Yfiltered)
# plt.figure()
# plt.imshow(Yfiltered[emwavelindices[0]:emwavelindices[1],timeminplot:timemaxplot],extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths[emwavelindices[0]:emwavelindices[1]]),np.min(wavelengths[emwavelindices[0]:emwavelindices[1]])],aspect='auto')#,vmax=1000)
# im0=ax[0].imshow(Yfiltered,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths),np.min(wavelengths)],aspect='auto')
im0=ax[0].imshow(Yfiltered[emwavelindices[0]:emwavelindices[1],:],extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths[emwavelindices[0]:emwavelindices[1]]),np.min(wavelengths[emwavelindices[0]:emwavelindices[1]])],aspect='auto',vmin=0)
# im0=ax[0].imshow(Yfiltered,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths),np.min(wavelengths)])
ax[0].set_ylabel('Emission wavelength (nm)',fontsize=16)
ax[0].yaxis.set_tick_params(labelsize=16)
# ax[0].set_xlabel('Time (s)')
# ax[0].set_title('Emission')
# ax[0].xaxis.set_tick_params(which='both', labelbottom=True)
# ax[0].xaxis.set_tick_params(which='both', labelbottom=True)
ax[0].invert_yaxis()
fig.colorbar(im0,ax=ax[0])
# vmax=50
vmax=np.max(excitationdata)
im1=ax[1].imshow(excitationdatacorr,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(excitationdata1[1]),np.min(excitationdata1[1][excitationdata1[1]>0])],aspect='auto',vmin=0,vmax=vmax)
# im1=ax[1].imshow(excitationdata[0],extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(excitationdata[1]),np.min(excitationdata[1][excitationdata[1]>0])])
ax[1].set_ylabel('Excitation wavelength (nm)',fontsize=16)
ax[1].set_xlabel('Time (s)',fontsize=16)
ax[1].xaxis.set_tick_params(labelsize=16)
ax[1].yaxis.set_tick_params(labelsize=16)
# ax[1].set_title('Excitation')
fig.colorbar(im1,ax=ax[1])
ax[1].invert_yaxis()
# plt.tight_layout()


#select excitation spectra based on trigger events from first photons in the emission cycle
#can do excitation emission correlation only on those. Forward and backward sweep combined



#%% excitation and emission correlation
taus=[0,1,2]

# plot = 'cov'
plot = 'corr'
# plot='all'
# plot = 'none'


emwavelrange=(590,630)

emwavelindices=(np.argmin(np.abs(wavelengths-emwavelrange[0])),np.argmin(np.abs(wavelengths-emwavelrange[1])))

emissionwavelengths = wavelengths[emwavelindices[0]:emwavelindices[1]]


# excwavelrange=(430,590)
excwavelrange=(505,575)
# excwavelrange=matchrange
excwavelindices=(np.argmin(np.abs(excitationdata1[1][:,1]-excwavelrange[0])),np.argmin(np.abs(excitationdata1[1][:,1]-excwavelrange[1])))

excitationwavelengths = excitationdata1[1][excwavelindices[0]:excwavelindices[1],1]

limlow = 600 #timeframes (in terms of their index. Can be visualized when seeing the plot produced in the section 'excitation and emission together')
limhigh = 700
Excitationdata = excitationdata[excwavelindices[0]:excwavelindices[1],limlow:limhigh]
Emissiondata = Yfiltered[emwavelindices[0]:emwavelindices[1],limlow:limhigh]-np.min(Yfiltered)


# excemmcovariance,excemmnormalization,excemmcorrelation = rplm.pearsoncorrelation(Emissiondata,Excitationdata,excitationwavelengths,emissionwavelengths,taus=taus,plot=plot)
# excemmcovariance,excemmnormalization,excemmcorrelation = rplm.spectralcorrelation(Emissiondata,Excitationdata,excitationwavelengths,emissionwavelengths,taus=taus,plot=plot)
excitationcovariance,excitationnormalization,excitationcorrelation = rplm.pearsoncorrelation(Excitationdata,Excitationdata,excitationwavelengths,excitationwavelengths,taus=taus,plot=plot)
# emissioncovariance,emissionnormalization,emissioncorrelation = rplm.pearsoncorrelation(Emissiondata,Emissiondata,emissionwavelengths,emissionwavelengths,taus=taus,plot=plot)
# emissioncovarance,emissionnormalization,emissioncorrelation = rplm.spectralcorrelation(Emissiondata,Emissiondata,emissionwavelengths,emissionwavelengths,taus=taus,plot=plot)
#%%
import matplotlib
from mpl_toolkits.axes_grid1 import AxesGrid
def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero.

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower offset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax / (vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highest point in the colormap's range.
          Defaults to 1.0 (no upper offset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)
    
        # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap



#%%
from matplotlib import colors


minwavspec1 = np.min(np.delete(excitationwavelengths.ravel(),np.where(excitationwavelengths.ravel()<=1))) #done to select the wavelength on the correct axis
maxwavspec1 = np.max(np.delete(excitationwavelengths.ravel(),np.where(excitationwavelengths.ravel()<=1))) # I dont want to construct a map with all zeros
minwavspec2 = np.min(np.delete(excitationwavelengths.ravel(),np.where(excitationwavelengths.ravel()<=1)))
maxwavspec2 = np.max(np.delete(excitationwavelengths.ravel(),np.where(excitationwavelengths.ravel()<=1)))

for i in range(len(taus)):
    tau=taus[i]

    # if tau==0:
    vmax0corr=np.nanmax(np.delete(excitationcorrelation.ravel(),np.where(excitationcorrelation.ravel()>=0.95))) #done to remove the whole =1 diagonal visualization and allows for better visualization of the plot
    # print(vmax0corr)
    plt.figure()
    # plt.imshow(excitationcorrelation[i], interpolation="none", cmap=shifted_cmap,extent=[minwavspec1,maxwavspec1,maxwavspec2,minwavspec2],vmax=vmax0corr)
    plt.imshow(excitationcorrelation[i],cmap='seismic',extent=[minwavspec1,maxwavspec1,maxwavspec2,minwavspec2],vmax=vmax0corr)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Wavelength (nm)')
    plt.title('Correlation map. tau = '+str(taus[i]))
    colors.DivergingNorm(vmin=-0.4, vcenter=0, vmax=1)
#%% #select particular exctiation wavelength based on chosen range emission wavelengths
emwavelrange = np.array([(606.5,609.5),(611.5,614.5),(615,618)])
emwavelrange = np.array([(595,600),(601,606)])
# emwavelrange = np.array([(518,521),(523,527)])
plt.figure()
plt.plot(np.sum(Yfiltered,axis=0))
intensitymaxoff=42000#base on sum Yfiltered plot
# emwavelrange = np.array([(600,605),(605,610)])
emwavindices = np.zeros(emwavelrange.shape)
for j in range(len(emwavelrange)):
    emwavindices[j] = (np.argmin(np.abs(wavelengths-emwavelrange[j,0])),np.argmin(np.abs(wavelengths-emwavelrange[j,1])))
emwavsum = np.zeros((len(Yfiltered[0]),len(emwavelrange)))
# emwavsum = np.zeros((len(emwavelrange),len(Yfiltered)))
for l in range(len(Yfiltered[0])):
    for j in range(len(emwavelrange)):
        emwavsum[l,j] = np.sum(Yfiltered[int(emwavindices[j,0]):int(emwavindices[j,1]),l])

maxindex = np.zeros(len(Yfiltered[0])) 
timesmaxindex = np.zeros(emwavsum.shape)
for j in range(len(Yfiltered[0])):
    maxindex[j] = np.argmax(emwavsum[j]) #identified maxima in spectra according to 4 states
    
emspecex=np.zeros(Yfiltered.shape)
emspectr=np.zeros(Yfiltered.shape)
emspecoff=np.zeros(Yfiltered.shape)

exspecex=np.zeros(excitationdata.shape)
exspectr=np.zeros(excitationdata.shape)
exspecoff=np.zeros(excitationdata.shape)

nrex=0
nrtr=0
nroff=0

tempsum=np.sum(Yfiltered,axis=0)
for j in range(len(Yfiltered[0])):
    if tempsum[j]>intensitymaxoff:
        if maxindex[j]==0:
            nrex+=1
            emspecex[:,j]=Yfiltered[:,j]
            exspecex[:,j]=excitationdata[:,j]
        elif maxindex[j]==1:
            nrtr+=1
            emspectr[:,j]=Yfiltered[:,j]
            exspectr[:,j]=excitationdata[:,j]
    else:
        nroff+=1
        emspecoff[:,j]=Yfiltered[:,j]
        exspecoff[:,j]=excitationdata[:,j]

# plt.figure() #unnormalized for laser
# plt.plot(wavelengths,np.sum(emspecex,axis=1)/nrex,label='exciton')        
# # plt.figure()
# # plt.figure()
# plt.plot(wavelengths,np.sum(emspectr,axis=1)/nrtr,label='trion')
# plt.title('emission')
# plt.legend()
# plt.figure()
# plt.plot(excitationdata[1][:,1],np.sum(exspecex,axis=1)/nrex,label='exciton')
# plt.plot(excitationdata[1][:,1],np.sum(exspectr,axis=1)/nrtr,label='trion')
# plt.title('excitation')
# plt.legend()
fig,ax=plt.subplots(2,1,sharex=True,sharey=False,constrained_layout=True) #plot excitation and emission above each other. If you want them to be visualized with the same window the yrange should cover the same distance. also zooming goes automatically
# emwavelrange=(500,540)
emwavelrange=(590,620)
emwavelindices=(np.argmin(np.abs(wavelengths-emwavelrange[0])),np.argmin(np.abs(wavelengths-emwavelrange[1])))
timeminplot=0
timemaxplot=-1
vmax=2000
vmax=np.max(Yfiltered[Yfiltered<np.max(Yfiltered)])
vmax=np.max(emspecex)
# plt.figure()
# plt.imshow(Yfiltered[emwavelindices[0]:emwavelindices[1],timeminplot:timemaxplot],extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths[emwavelindices[0]:emwavelindices[1]]),np.min(wavelengths[emwavelindices[0]:emwavelindices[1]])],aspect='auto')#,vmax=1000)
# im0=ax[0].imshow(Yfiltered,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths),np.min(wavelengths)],aspect='auto')
im0=ax[0].imshow(emspecex,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths[emwavelindices[0]:emwavelindices[1]]),np.min(wavelengths[emwavelindices[0]:emwavelindices[1]])],aspect='auto',vmin=0)
# im0=ax[0].imshow(Yfiltered,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths),np.min(wavelengths)])
ax[0].set_ylabel('Emission wavelength (nm)')
ax[0].set_xlabel('Time (s)')
ax[0].set_title('Emission')
ax[0].xaxis.set_tick_params(which='both', labelbottom=True)
ax[0].invert_yaxis()
fig.colorbar(im0,ax=ax[0])

im0=ax[1].imshow(emspecoff,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths[emwavelindices[0]:emwavelindices[1]]),np.min(wavelengths[emwavelindices[0]:emwavelindices[1]])],aspect='auto',vmin=0)
# im0=ax[0].imshow(Yfiltered,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths),np.min(wavelengths)])
ax[1].set_ylabel('Emission wavelength (nm)')
ax[1].set_xlabel('Time (s)')
ax[1].set_title('Emission')
ax[1].xaxis.set_tick_params(which='both', labelbottom=True)
ax[1].invert_yaxis()
fig.colorbar(im0,ax=ax[1])
plt.figure()
plt.plot(wavelengths,np.sum(emspecex,axis=1)/np.sum(emspecex),label='Bright')        
plt.plot(wavelengths,np.sum(emspectr,axis=1)/np.sum(emspectr),label='Gray')
# plt.plot(wavelengths,np.sum(emspecoff,axis=1)/np.sum(emspecoff),label='Dark')
plt.title('Emission')
plt.ylabel('Intensity')
plt.xlabel('Emission wavelength (nm)')
plt.legend()


plt.figure()
plt.plot(excitationdata1[1][:,1],np.sum(exspecex,axis=1)/np.sum(exspecex)/(interprefspec(excitationdata1[1][:,1])),label='Bright')
plt.plot(excitationdata1[1][:,1],np.sum(exspectr,axis=1)/np.sum(exspectr)/interprefspec(excitationdata1[1][:,1]),label='Gray')
# plt.plot(excitationdata1[1][:,1],np.sum(exspecoff,axis=1)/np.sum(exspecoff)/interprefspec(excitationdata1[1][:,1]),label='Dark')
plt.title('Excitation')
plt.ylabel('Intensity')
plt.xlabel('Excitation wavelength (nm)')
plt.legend(loc=0)
# plt.xlim(500,575)
# plt.ylim(0,0.01)

#%% Lifetime vs Intensity
#show the corresponding lifetime traces based on emission wavelength selection
# MaxLikelihoodFunction_c = nb.jit(nopython=True)(MaxLikelihoodFunction)
macrotimesin=data[3]
microtimesin=data[2]
# binwidth=0.01
macrolimits = rpl.HistPhotons(macrotimesin*dtmacro,binwidth,Texp)

macrotimesin=data[25]
macrolimits=macrotimesin

limits=macrolimits






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

    if maxindex[tbinnr]==0 and tempsum[tbinnr]>intensitymaxoff:# and taulist[tbinnr]<lifetimeexciton:

        microtimes_ex[nrex:nrex+nphots]=microtimes
        macrotimescycle_ex[nrex:nrex+nphots]=macrotimescycle
        bins_ex+=1
        nrex+=nphots
        timeexciton[tbinnr]=1


    # elif nphots>limittrlow and nphots<limittrhigh and taulist[tbinnr]<10 and tauav[tbinnr]>1:# and taulist[tbinnr]>0.05: #and (photonspbin[tbinnr]-7)/taulist[tbinnr]>112/28
        # print('found')

    elif maxindex[tbinnr]==1 and tempsum[tbinnr]>intensitymaxoff:
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
    elif tempsum[tbinnr]<intensitymaxoff:
        microtimes_off[nroff:nroff+nphots]=microtimes
        macrotimescycle_off[nroff:nroff+nphots]=macrotimescycle
        bins_off+=1
        nroff+=nphots

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
fitex=rpl.GetLifetime(microtimes_ex,dtmicro,dtmacro,200e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_ex/(Texp/binwidth),expterms=2,method='ML_c')
# fitex=GetLifetime(microtimes_ex,dtmicro,dtmacro,100e-9,tstart=-1,plotbool=True,ybg=bins_ex*40*binwidth/np.max(microtimesblue),method='ML_c')
fittrion=rpl.GetLifetime(microtimes_trion,dtmicro,dtmacro,5e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_trion/(Texp/binwidth),expterms=2,method='ML_c')
# fittrion=GetLifetime(microtimes_trion,dtmicro,dtmacro,5e-9,tstart=-1,plotbool=True,ybg=bins_trion*40*binwidth/np.max(microtimesblue),method='ML_c')

fitoff=rpl.GetLifetime(microtimes_off,dtmicro,dtmacro,10e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_trion/(Texp/binwidth),method='ML_c')
# fitmid=rpl.GetLifetime(microtimes_off,dtmicro,dtmacro,200e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_mid/(Texp/binwidth),expterms=2,method='ML_c')

print('Rad lifetime ratio:'+str(fittrion[1]/bins_trion/(fitex[1]/bins_ex)))
#plt.xlim([0,220])
# plt.legend(['High cps','High cps fit','Mid cps','Mid cps fit','Low cps','Low cps fit'])


plt.figure()
plt.semilogy(fitex[4],fitex[5])
plt.semilogy(fittrion[4],fittrion[5])
plt.semilogy(fitoff[4],fitoff[5])
plt.title('Time-averaged lifetimes')
plt.xlabel('time (ns)')
plt.ylabel('counts (a.u.)')

#%% Excitation overplot with emission fitted
emfit=rplm.fitspectra(Yfiltered,wavelengths,0,len(Yfiltered[0]),'Lor',Debugmode=False)

plt.figure()
plt.imshow(Yfiltered,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,wavelengths[-1],wavelengths[0]],vmin=0,aspect='auto')
plt.plot(emfit[2],'*',c='r')
plt.gca().invert_yaxis()
plt.colorbar()
plt.ylabel('Emission Wavelength (nm)')
plt.xlabel('time (s)')
plt.title('Full emission spectra')  

plt.figure()
plt.imshow(Yfiltered,extent=[0,len(Yfiltered[0]),wavelengths[-1],wavelengths[0]],vmin=0,aspect='auto')
plt.plot(emfit[2],'*',c='r')
plt.gca().invert_yaxis()
plt.colorbar()
plt.ylabel('Emission Wavelength (nm)')
plt.xlabel('time (s)')
plt.title('Full emission spectra') 

fig,ax=plt.subplots(nrows=1,ncols=2)
ax[0].imshow(excitationdata,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,wavelengthrange[1],wavelengthrange[0]],vmin=0,vmax=vmax,aspect='auto')
ax[1].plot()
plt.gca().invert_yaxis()
plt.colorbar()
plt.ylabel('Excitation Wavelength (nm)')
plt.xlabel('time (s)')
plt.title('Full excitation spectra')  

fig,ax1 = plt.subplots()
ax1.imshow(excitationdata,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,wavelengthrange[1],wavelengthrange[0]],vmin=0,vmax=vmax,aspect='auto')
plt.gca().invert_yaxis()
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Excitation wavelength (nm)', color='b')
ax1.tick_params('y', colors='b')
ax2 = ax1.twinx()
ax2.scatter(np.linspace(data[26][0]*dtmacro,data[26][-1]*dtmacro,len(data[26])),emfit[2],'r')
ax2.set_ylabel('fitted emmission maximum (ns)', color='r')
ax2.tick_params('y', colors='r')




#%% correlation of fits of spectra
#just 1D correlation based on fitted maxima to rule out ampltidue fluctuations and see how the peaks moved in time. Almost always a correlation behavriour was observed for all the dots. Also for synthetica data
fitted=rplm.fitspectra(Emissiondata,emissionwavelengths,0,100,model='Lor',Debugmode=False)
fitted=rplm.fitspectra(Emissiondata,emissionwavelengths,0,100,model='Lor',Debugmode=False)

# fitted[0][51]=fitted[0][50]
plt.figure()
plt.plot(fitted[0])
plt.xlabel('time')
plt.ylabel('Center wavelength (nm)')
plt.title('Fitted amplitudes')

fittedmean = rplm.pearsoncorrelation1D(fitted[0])
plt.figure()
plt.scatter(fittedmean[3],fittedmean[0])
plt.plot(fittedmean[3],np.repeat(-fittedmean[2],len(fittedmean[0])),c='r');plt.plot(fittedmean[3],np.repeat(fittedmean[2],len(fittedmean[0])),c='r');
plt.xlabel('Delay tau')
plt.ylabel('Correlation')
plt.title('Correlation of fitted amplitudes')

plt.figure()
plt.plot(fitted[2])
plt.xlabel('time')
plt.ylabel('Center wavelength (nm)')
plt.title('Fitted maxima')

fittedmean = rplm.pearsoncorrelation1D(fitted[2][:150])
plt.figure()
plt.scatter(fittedmean[3],fittedmean[0])
plt.plot(fittedmean[3],np.repeat(-fittedmean[2],len(fittedmean[0])),c='r');plt.plot(fittedmean[3],np.repeat(fittedmean[2],len(fittedmean[0])),c='r');
plt.xlabel('Delay tau')
plt.ylabel('Correlation')
plt.title('Correlation of fitted maxima')

# fitted[4][51]=fitted[4][50]
plt.figure()
plt.plot(fitted[4])
plt.xlabel('time')
plt.ylabel('Width (nm)')
plt.title('Fitted widths')

fittedmean = rplm.pearsoncorrelation1D(fitted[4])
plt.figure()
plt.plot(fittedmean[3],fittedmean[0])
plt.plot(fittedmean[3],np.repeat(-fittedmean[2],len(fittedmean[0])),c='r');plt.plot(fittedmean[3],np.repeat(fittedmean[2],len(fittedmean[0])),c='r');
plt.xlabel('Delay tau')
plt.ylabel('Correlation')
plt.title('Correlation of fitted widths')


    #%%     Fourier transform the correlation data

guessorigin=524
origin,rest = rplm.find_origin(emissionnormalization[1],guessorigin,emissionwavelengths,prominence=1,width=5) #find origin works based on the normalization map. It is semidieal, should definitely check whether the emission corelation map shows the same origin as was produced by thsi function

# origin= 24
# print(testwavelengths[origin])
datapoints = 40 #datapoints can not be an odd number
# origin= 129
# origin = np.where(emissioncovariance==np.amax(emissioncovariance))[2][0]  #because it is data which is sorted like : taus, wavelengths, wavelengths
print('polar transform centered around wavelength (nm)',emissionwavelengths[origin])
datapoints = 26

correlationpolar = rplm.cartesiantopolarcorrelation(emissioncorrelation,origin,datapoints)
if Debugmode==True:
    plt.figure()
    # plt.imshow(correlationpolar[1][1])
    plt.imshow(correlationpolar[1][1],extent=[np.min(correlationpolar[2]*180/np.pi),np.max(correlationpolar[2]*180/np.pi),int(datapoints/np.sqrt(2)),0],aspect='auto')
    plt.colorbar()
    plt.ylabel('Radius (pixel)')
    plt.xlabel('Angle (degrees)')       
    plt.title('Polar transform correlation map')

correlationpolarsum=np.sum(correlationpolar[1],axis=1)
correlationpolardata,covariancethetas = correlationpolar[1],correlationpolar[2] #use cartesian data as input

fourierdata = rplm.fouriertransform(correlationpolardata,covariancethetas)
fourierdata =fourierdata
fourierthetashift = fourierdata[2] #is shifted to make the plot look cleaner
fourierdatareal = fourierdata[3] # is shifted the same way as theta
fourierdataimag = fourierdata[4]




#%% Plot fourier transforms
selectedradii = [5,10,20,30]
# selectedradii= np.arange(0,10)
selectedradii= np.arange(0,len(fourierdata[3][1]))
# selectedradii = [40] # select a particular radius
# fourierdataradius =
if Debugmode==True: #plots the individual fourier transforms
    # for j in range(len(fourierdata[0])):
    for j in range(0,1):
    # for j in range(0,101,20):
        tau = taus[j]
        plt.figure()
        spectrum = j # this parameter is for different taus

        for i in range(len(selectedradii)):
        # for i in range(len(fourierdatareal[0])):
            selectedradius = selectedradii[i]
        # for i in range(20):
            # plt.plot(np.fft.fftshift(fouriertautheta[j][i]),np.fft.fftshift(fouriertaudata[spectrum][i]).imag)#,label=i) #take imaginairy component because the data is assymetric I think
            plt.plot(fourierthetashift[spectrum][selectedradius],fourierdatareal[spectrum][selectedradius],label='R = '+str(selectedradius)+' pix') 
            # plt.plot(fourierthetashift[spectrum][i],fourierdataimag[spectrum][i]) 
        
            # plt.ylim(0,20)
            plt.xlabel('n-fold symmetry')
            plt.ylabel('Frequency Domain (Spectrum) Magnitude')
            plt.title('Fourier Transform real. Tau = ' + str(tau)+' Wav. Org. = ' +str(testwavelengths[origin])[:5]+' nm')
            plt.xlim(-6,6)
            plt.legend()
    # 
    # for j in range(len(fourierdata[0])):
    
    for j in range(0,1):
        tau = taus[j]
        plt.figure()
        spectrum = j # this parameter is for different taus

        # for i in range(len(selectedradii)):
        for i in range(0,20):
            
        # for i in range(len(fourierdatareal[0])):
            selectedradius = selectedradii[i]
        # for i in range(20):
            # plt.plot(np.fft.fftshift(fouriertautheta[j][i]),np.fft.fftshift(fouriertaudata[spectrum][i]).imag)#,label=i) #take imaginairy component because the data is assymetric I think
            # plt.plot(fourierthetashift[spectrum][i],fourierdatareal[spectrum][i]) 
            plt.plot(fourierthetashift[spectrum][selectedradius],fourierdataimag[spectrum][selectedradius],label='R = '+str(selectedradius)+' pix') 
            # plt.plot(fourierthetashift[spectrum][i],fourierdataimag[spectrum][i]) 
        
            # plt.ylim(0,20)
            plt.xlabel('n-fold symmetry')
            plt.ylabel('Frequency Domain (Spectrum) Magnitude')
            plt.title('Fourier Transform imag. Tau = ' + str(tau) +' Wav. Org. = ' +str(emissionwavelengths[origin])[:5]+' nm')
            plt.xlim(-6,6)
            plt.legend(loc=1)

#here you can see how the amplitude of a particular n-fold cycle displays vs tau. Note that when you have a single tau defined you will get an empty plot (because there is only one point)

ftimag = rplm.Fouriercomponentvstau(fourierdataimag, fourierthetashift, selectedradii)
ftreal = rplm.Fouriercomponentvstau(fourierdatareal, fourierthetashift, selectedradii)

if Debugmode==True: #these plots are most definitely the most interesting ones as the 2D data is shown versuss delay time and radius (with the color the amplitude of the component)
    plt.figure()
    plt.imshow(ftimag[1],extent=[np.min(selectedradii),np.max(selectedradii),np.max(taus),np.min(taus)])
    plt.xlabel('Radius')
    plt.ylabel('Delay tau')
    plt.title('Zeroth imaginairy component')
    
    plt.figure()
    plt.imshow(-ftimag[5],extent=[np.min(selectedradii),np.max(selectedradii),np.max(taus),np.min(taus)])
    plt.xlabel('Radius')
    plt.ylabel('Delay tau')
    plt.title('Second imaginairy component')
    plt.colorbar()
    

if Debugmode==True: #same but for the real components
    plt.figure()
    plt.imshow(ftreal[1])
    plt.xlabel('radius')
    plt.ylabel('tau')
    plt.title('Zeroth component real')
    
    plt.figure()
    plt.imshow(ftreal[5])
    plt.xlabel('radius')
    plt.ylabel('tau')
    plt.title('Second component real')

#%% G2
# MakeG2_c=nb.jit(nopython=True)(rpl.MakeG2)
# plt.figure()
g2=rplm.MakeG2(times0,times1,dtmicro,g2restime=dtmicro*4,nrbins=1000)

# timemin=395
# timemax=420
# indices0=(np.argmin((times0*dtmicro-timemin)**2),np.argmin((times0*dtmicro-timemax)**2))
# indices1=(np.argmin((times1*dtmicro-timemin)**2),np.argmin((times1*dtmicro-timemax)**2))

# g2=rplm.MakeG2(times0[indices0[0]:indices0[1]],times1[indices1[0]:indices1[1]],dtmicro,g2restime=dtmicro*4,nrbins=1000)
# plt.figure()
# plt.plot(g2[0],g2[1])
#%% efficiency biexciton g2
#no exciton at zero delay
timeindices0=(-25,25)
idxindices0=(np.argmin(np.abs(g2[0]-timeindices0[0])),np.argmin(np.abs(g2[0]-timeindices0[1])))
sumbiexciton=np.sum(g2[1][idxindices0[0]:idxindices0[1]])
timeindices1=(66,140)
idxindices1=(np.argmin(np.abs(g2[0]-timeindices1[0])),np.argmin(np.abs(g2[0]-timeindices1[1])))
sumexciton=np.sum(g2[1][idxindices1[0]:idxindices1[1]])

biexcitonQY=sumbiexciton/sumexciton
print('biexcitonQY = ',biexcitonQY)
#%% laser corrected g2
histsettings=(440,600,200)
maxwavel=580
indices0=np.argwhere((calibcoeffs[1]+rplm.InVoltagenew(data[19]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0])<maxwavel)
exspec=rpl.Easyhist(calibcoeffs[1]+rplm.InVoltagenew(data[19][indices0]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
# plt.plot(exspec[0],(exspec[1]-np.min(exspec[1]))/np.sum(exspec[1]-np.min(exspec[1])))
indices1=np.argwhere((calibcoeffs[1]+rplm.InVoltagenew(data[20]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0])<maxwavel)
g2data=rplm.MakeG2(times0[indices0],times1[indices1],dtmicro,g2restime=dtmicro*4,nrbins=1000)
# plt.figure()
# plt.plot(g2data[0],g2data[1])

maxtime=-3 #in ns
indices0=np.argwhere((data[2]-lifetime[3])*dtmicro*1e9-1>maxtime)
indices1=np.argwhere((data[4]-lifetime[3])*dtmicro*1e9-1>maxtime)


#%% laser and time gated g2
maxtime=5

maxwavel=580

indices0=np.argwhere((data[2]-lifetime[3])*dtmicro*1e9-1>maxtime).ravel()
indices0a=np.argwhere((calibcoeffs[1]+rplm.InVoltagenew(data[19][indices0]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0])<maxwavel).ravel()

# plt.plot(exspec[0],(exspec[1]-np.min(exspec[1]))/np.sum(exspec[1]-np.min(exspec[1])))
indices1=np.argwhere((data[4]-lifetime[3])*dtmicro*1e9-1>maxtime).ravel()
indices1a=np.argwhere((calibcoeffs[1]+rplm.InVoltagenew(data[20][indices1]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0])<maxwavel).ravel()

g2data=rplm.MakeG2(times0[indices0[indices0a]],times1[indices1[indices1a]],dtmicro,g2restime=dtmicro*4,nrbins=1000)
# plt.figure()
# plt.plot(g2data[0],g2data[1])


#%% G2 for biexciton, mirotime-gated

maxtime=5 #in ns
indices0=np.argwhere((data[2]-lifetime[3])*dtmicro*1e9-1>maxtime)
indices1=np.argwhere((data[4]-lifetime[3])*dtmicro*1e9-1>maxtime)
# exspec=rpl.Easyhist(calibcoeffs[1]+rplm.InVoltagenew(data[19][indices0]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
# plt.plot(exspec[0],(exspec[1]-np.min(exspec[1]))/np.sum(exspec[1]-np.min(exspec[1])))
# indices1=np.argwhere((calibcoeffs[1]+rplm.InVoltagenew(data[20]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0])<580)
g2data=rplm.MakeG2(times0[indices0],times1[indices1],dtmicro,g2restime=dtmicro*4,nrbins=1000)
# g2data=rplm.MakeG2(times0,times1,dtmicro,g2restime=dtmicro*4,nrbins=1000)
# plt.figure()
# plt.plot(g2data[0],g2data[1])
# plt.xlabel('delay (ns)')
# plt.ylabel('occurence (a.u.)')
# plt.title('g(2) correlation')

timeindices0=(-25,25)
idxindices0=(np.argmin(np.abs(g2data[0]-timeindices0[0])),np.argmin(np.abs(g2data[0]-timeindices0[1])))
sumbiexciton=np.sum(g2data[1][idxindices0[0]:idxindices0[1]])
timeindices1=(66,140)
idxindices1=(np.argmin(np.abs(g2data[0]-timeindices1[0])),np.argmin(np.abs(g2data[0]-timeindices1[1])))
sumexciton=np.sum(g2data[1][idxindices1[0]:idxindices1[1]])

biexcitonQY=sumbiexciton/sumexciton
print('biexcitonQY = ',biexcitonQY)    

#%% Plot spectrum in eV
plt.figure()
excitationenergy = nmtoeV(Wavelengthspec)
plt.xlim((min(excitationenergy),max(excitationenergy)))
plt.plot(nmtoeV(Wavelengthspec),ylistspec/interprefspec(Wavelengthspec)/50000)
plt.ylim((0, 1.1))
#plt.xlim((445,535))
plt.xlabel('Excitation Energy - Emission Energy (eV)')
plt.ylabel('Intensity')
plt.title(namelist[0])



#%% Excitation resolved decay
plotrange=[500,590]
# plotrange=matchrange
# plotrange=excwavelrange
plt.figure()
mtimelims=np.array([0,30])
binning=1
mtimelimsdtmic=np.round(mtimelims/data[0]*1e-9)
nbins=mtimelimsdtmic[1]-mtimelimsdtmic[0]
#
histarray, Xedge, Yedge = np.histogram2d(data[2]*data[0]*1e9,calibcoeffs[1]+InVoltagenew_c(data[19]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],range=[[mtimelimsdtmic[0]*data[0]*1e9,mtimelimsdtmic[1]*data[0]*1e9],plotrange],bins=[int((nbins-1))*binning,50])
histarray = histarray.transpose() # tranpose somehow is necessary to plot this stuff right
normalization = np.sum(histarray,axis=0)
Xedge = Xedge[:-1]+0.5*(Xedge[1]-Xedge[0])
Yedge = Yedge[:-1]+0.5*(Yedge[1]-Yedge[0])
correction = np.zeros(histarray.shape)
for j in range(len(normalization)):
    correction[:,j] = histarray[:,j] / normalization[j]
plt.figure()
plt.imshow(histarray,aspect='auto',extent=[np.min(Xedge),np.max(Xedge),np.max(Yedge),np.min(Yedge)]) 
# plt.imshow(correction,aspect='auto',extent=[np.min(Xedge),np.max(Xedge),np.max(Yedge),np.min(Yedge)]) 
plt.xlabel('Time (ns)')
plt.ylabel('Excitation wavelength (nm)')
plt.title('Uncorrected')
plt.gca().invert_yaxis()
plt.colorbar()

if Debugmode==True:
    plt.figure()
    plt.imshow(correction,aspect='auto',extent=[np.min(Xedge),np.max(Xedge),np.max(Yedge),np.min(Yedge)]) 
    # plt.imshow(correction,aspect='auto',extent=[np.min(Xedge),np.max(Xedge),np.max(Yedge),np.min(Yedge)]) 
    plt.xlabel('Time (ns)')
    plt.ylabel('Excitation wavelength (nm)')
    plt.gca().invert_yaxis()
    plt.title('timenormalized')
    plt.colorbar()

#actually the only difference with normalization or not is that the intensity at lower times are more/less pronounced



#%% Make excitation wavelength resolved decay plot
plt.figure()

mtimelims=np.array([0,100])
binning=1
mtimelimsdtmic=np.round(mtimelims/data[0]*1e-9)
nbins=mtimelimsdtmic[1]-mtimelimsdtmic[0]
plt.hist2d(data[2]*data[0]*1e9,calibcoeffs[1]+InVoltage(data[19]*data[1],Freq,Vpp,Voffset,Verror)*calibcoeffs[0],range=[[int(mtimelimsdtmic[0])*data[0]*1e9,int(mtimelimsdtmic[1])*data[0]*1e9],[530,590]],bins=[(int(nbins)-1)*binning,50])#,norm=mcolors.LogNorm())

plt.xlabel('Time (ns)')
plt.ylabel('Excitation wavelength (nm)')
plt.colorbar()
#%% Ex spectrum vs time
plt.figure()
plt.hist2d(data[3]*data[1],calibcoeffs[1]+InVoltagenew_c(data[19]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],range=[[1,Texp],matchrange],bins=[120,160])#,norm=mcolors.LogNorm())
plt.xlabel('Time (s)')
plt.ylabel('Excitation wavelength (nm)')
plt.colorbar()

#%% Excitation gate delay time
microtimesred = np.zeros(len(data[2]),dtype='int64')
macrotimesred = np.zeros(len(data[2]),dtype='int64')
nrred = 0
microtimesblue = np.zeros(len(data[2]),dtype='int64')
macrotimesblue = np.zeros(len(data[2]),dtype='int64')
macrotimescycleblue = np.zeros(len(data[2]),dtype='int64')
timesblue = np.zeros(len(data[2]),dtype='int64')
nrblue = 0

wavellimit = 590
wavellimitlow= 500


exwavel = calibcoeffs[1]+InVoltagenew_c(data[19]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0]
for j in tqdm(range(len(data[2]))):
    if exwavel[j] > wavellimit:
        microtimesred[nrred] = data[2][j]
        macrotimesred[nrred] = data[3][j]
        nrred += 1
    elif exwavel[j] > wavellimitlow:
        microtimesblue[nrblue] = data[2][j]
        macrotimesblue[nrblue] = data[3][j]
        macrotimescycleblue[nrblue] = data[19][j]
        timesblue[nrblue]=times0[j]
        nrblue += 1
microtimesred = microtimesred[:nrred]
macrotimesred = macrotimesred[:nrred]
microtimesblue = microtimesblue[:nrblue]
macrotimesblue = macrotimesblue[:nrblue]
macrotimescycleblue = macrotimescycleblue[:nrblue]



#%% Plot excitation gated decay
# plt.figure()
MaxLikelihoodFunction_c = nb.jit(nopython=True)(rpl.MaxLikelihoodFunction)
fitdata=rpl.GetLifetime(microtimesblue,dtmicro,dtmacro,250e-9,tstart=-1,plotbool=True,ybg=-1,method='ML_c')
# plt.xlim([18,100])
# plt.set_yscale('log')

#%% Lifetime vs Intensity
# MaxLikelihoodFunction_c = nb.jit(nopython=True)(MaxLikelihoodFunction)
macrotimesin=data[3]
microtimesin=data[2]
# binwidth=0.01
macrolimits = rpl.HistPhotons(macrotimesin*dtmacro,binwidth,Texp)

macrotimesin=data[25]
macrolimits=macrotimesin

limits=macrolimits
taulist=np.zeros(len(limits)-1)
taulist1=np.zeros(len(limits)-1)
taulist2=np.zeros(len(limits)-1)
tauav=np.zeros(len(limits)-1)
Alist=np.zeros(len(limits)-1)
Alist1=np.zeros(len(limits)-1)
Alist2=np.zeros(len(limits)-1)
photonspbin=np.zeros(len(limits)-1)
photonspbin=np.zeros(len(limits)-1)
ybglist=np.zeros(len(limits)-1)
buff=np.zeros(len(limits)-1)
meanex=np.zeros(len(limits)-1)
stdex=np.zeros(len(limits)-1)
histbinmultiplier=1
plt.figure()
lifetime1=rpl.GetLifetime(data[4],dtmicro,dtmacro,250e-9,tstart=-1,histbinmultiplier=1,ybg=0,plotbool=True,expterms=2,method='ML_c')
plt.show()

for tbinnr in tqdm(range(len(limits)-1)):
    microtimes = microtimesin[limits[tbinnr]:limits[tbinnr+1]]

    [taulist[tbinnr],Alist[tbinnr],ybglist[tbinnr],buff[tbinnr],temp,temp2]=rpl.GetLifetime(microtimes,dtmicro,dtmacro,300e-9,tstart=lifetime[3]*dtmicro,histbinmultiplier=1,ybg=lifetime[2]*binwidth/Texp,plotbool=False,expterms=1,method='ML_c') 
    # lifetimetot=rpl.GetLifetime(microtimes,dtmicro,dtmacro,300e-9,tstart=lifetime[3]*dtmicro,histbinmultiplier=1,ybg=lifetime[2]*binwidth/Texp,plotbool=False,expterms=2,method='ML_c') 
    # [taulist1[tbinnr],taulist2[tbinnr],Alist1[tbinnr],Alist2[tbinnr],ybglist[tbinnr],buff[tbinnr]]=[lifetimetot[0][0],lifetimetot[0][1],lifetimetot[1][0],lifetimetot[1][1],lifetimetot[2],lifetimetot[3]]
    tauav[tbinnr]=(np.mean(microtimes)-lifetime[3])*dtmicro*1e9-1
    photonspbin[tbinnr]=len(microtimes)

if len(limits)==len(data[25]):
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
    fig,ax1 = plt.subplots()
    ax1.plot(data[26][:-1]*dtmacro,photonspbin,'b')
    #ax1.plot(macrotimesin[limits[0:len(limits)-1]]*dtmacro,Alist,'b')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Photons per emission bin', color='b')
else:
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
    
    fig,ax1 = plt.subplots()
    ax1.plot(macrotimesin[limits[0:len(limits)-1]]*dtmacro,photonspbin,'b')
    #ax1.plot(macrotimesin[limits[0:len(limits)-1]]*dtmacro,Alist,'b')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Photons per '+str(int(binwidth*1000))+' ms bin', color='b')


#%% collective lifetimes without assumption of states and thresholds
    #just pick individual states and plot the corresponding spectra
beginbins=[169,252,317]    
endbins=[176,270,350]    
macrotimescyclein=data[2]

histsettings=(500,575,100)  


def timepostselectionlifetimes(beginbins,endbins,macrotimescyclein,microtimesin,limits,Debugmode=True):
 
    # macrotimescycle_ex= np.zeros(len(microtimesin),dtype='int64')
    if Debugmode==True:
        plt.figure()
    for idx in range(len(beginbins)):
        beginbin=beginbins[idx]
        endbin=endbins[idx]
        nrex=0
        bins_ex=0
        macrotimescycle_ex= np.zeros(len(microtimesin),dtype='int64')
        microtimes_ex= np.zeros(len(microtimesin),dtype='int64')
        
        for tbinnr in range(beginbin,endbin-1):
            nphots = limits[tbinnr+1]-limits[tbinnr]
            microtimes = microtimesin[limits[tbinnr]:limits[tbinnr+1]]
            macrotimescycle = macrotimescyclein[limits[tbinnr]:limits[tbinnr+1]]
            microtimes_ex[nrex:nrex+nphots]=microtimes
            macrotimescycle_ex[nrex:nrex+nphots]=macrotimescycle
            bins_ex+=1
            nrex+=nphots
    
        macrotimescycle_ex = macrotimescycle_ex[:nrex]
            
    # plt.figure()
        fitex=rpl.GetLifetime(microtimes_ex,dtmicro,dtmacro,200e-9,tstart=-1,plotbool=False,ybg=lifetime[2]*bins_ex/(Texp/binwidth),expterms=2,method='ML_c')
        fitex[5][0]=1
    # if Debugmode==True:
        # plt.figure()
        plt.semilogy(fitex[4],fitex[5],label='bins='+str(beginbin)+'-'+str(endbin))
        plt.title('Decay histograms')
        plt.xlabel('time (ns)')
        plt.ylabel('counts (a.u.)')
        plt.legend()
        
def timepostselectionexcitation(beginbins,endbins,microtimesin,limits,histsettings,Debugmode=False):
    
    if Debugmode==True:
        plt.figure()
    for idx in range(len(beginbins)):
        beginbin=beginbins[idx]
        endbin=endbins[idx]
        nrex=0
        bins_ex=0
        macrotimescycle_ex= np.zeros(len(microtimesin),dtype='int64')
        microtimes_ex= np.zeros(len(microtimesin),dtype='int64')
        for tbinnr in range(beginbin,endbin-1):
           nphots = limits[tbinnr+1]-limits[tbinnr]
           microtimes = microtimesin[limits[tbinnr]:limits[tbinnr+1]]
           macrotimescycle = macrotimescyclein[limits[tbinnr]:limits[tbinnr+1]]
       
           # if nphots>limitex:# and taulist[tbinnr]<lifetimeexciton:
       
           microtimes_ex[nrex:nrex+nphots]=microtimes
           macrotimescycle_ex[nrex:nrex+nphots]=macrotimescycle
           bins_ex+=1
           nrex+=nphots
        
        macrotimescycle_ex = macrotimescycle_ex[:nrex]

        macrotimescycle_ex = macrotimescycle_ex[:nrex]
        exspecex=rplm.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_ex*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
        exylistex=exspecex[1]/interprefspec(exspecex[0])
        plt.plot(exspecex[0],exylistex/np.sum(exylistex),label='bins='+str(beginbin)+'-'+str(endbin))

        plt.xlabel('Excitation wavelength')
        plt.ylabel('Intensity')
        plt.title('Excitation spectra')
        plt.legend()
        
def timepostselectionemission(beginbins,endbins,Debugmode=True):
    
    if Debugmode==True:
        plt.figure()
    for idx in range(len(beginbins)):
        beginbin=beginbins[idx]
        endbin=endbins[idx]
        # for tbinnr in range(beginbin,endbin-1):
            
        emdata=Yfiltered[:,beginbin:endbin]
        # plt.plot(excitationdata[:,1],np.sum(excitationdata,axis=1)/np.sum(exspecex)/(interprefspec(excitationdata1[1][:,1])),label='Bright')
        plt.plot(wavelengths,np.sum(emdata,axis=1)/np.sum(emdata),label='bins='+str(beginbin)+'-'+str(endbin))
        # plt.plot(wavelengths,np.sum(emspectr,axis=1)/np.sum(emspectr),label='Gray')
        # plt.plot(wavelengths,np.sum(emspecoff,axis=1)/interprefspec(wavelengths)/np.sum(emspecoff),label='Dark')
        # plt.plot(excitationdata1[1][:,1],np.sum(exspecoff,axis=1)/np.sum(exspecoff)/interprefspec(excitationdata1[1][:,1]),label='Dark')
        plt.title('Emission spectra')
        plt.ylabel('Intensity')
        plt.xlabel('Emission wavelength (nm)')
        # plt.xlim(590,620)
        plt.legend()
            
        
timepostselectionlifetimes(beginbins,endbins,macrotimescyclein,microtimesin,limits,Debugmode=True)
timepostselectionexcitation(beginbins,endbins,microtimesin,limits,histsettings,Debugmode=True)       #somehow the excitation spectra dont work properyl
timepostselectionemission(beginbins,endbins,Debugmode=True)     
    #%% individual but works properly
    
beginbin=175
endbin=190

beginbin=222
endbin=250

# beginbin=317
# endbin=350

macrotimescyclein=data[19]
microtimes_ex= np.zeros(len(data[2]),dtype='int64')
macrotimescycle_ex= np.zeros(len(data[2]),dtype='int64')
nrex=0
bins_ex=0

for tbinnr in range(beginbin,endbin-1):
    nphots = limits[tbinnr+1]-limits[tbinnr]
    microtimes = microtimesin[limits[tbinnr]:limits[tbinnr+1]]
    macrotimescycle = macrotimescyclein[limits[tbinnr]:limits[tbinnr+1]]

    # if nphots>limitex:# and taulist[tbinnr]<lifetimeexciton:

    microtimes_ex[nrex:nrex+nphots]=microtimes
    macrotimescycle_ex[nrex:nrex+nphots]=macrotimescycle
    bins_ex+=1
    nrex+=nphots

macrotimescycle_ex = macrotimescycle_ex[:nrex]
        
# plt.figure()
fitex=rpl.GetLifetime(microtimes_ex,dtmicro,dtmacro,200e-9,tstart=-1,plotbool=False,ybg=lifetime[2]*bins_ex/(Texp/binwidth),expterms=2,method='ML_c')
fitex[5][0]=1

# plt.figure()
# plt.semilogy(fitex[4],fitex[5])
# plt.title('bins='+str(beginbin)+'-'+str(endbin))
# plt.xlabel('time (ns)')
# plt.ylabel('counts (a.u.)')



histsettings=(500,575,100)
plt.figure()
exspecex=rplm.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_ex*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
exylistex=exspecex[1]/interprefspec(exspecex[0])
plt.plot(exspecex[0],exylistex/np.sum(exylistex),label='Bright')

# exspectr=rplm.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_trion*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
# exylisttr=exspectr[1]/interprefspec(exspectr[0])
# plt.plot(exspectr[0],exylisttr/np.sum(exylisttr),label='Gray')
# exspecoff=rplm.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_off*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
# plt.plot(exspecoff[0],exspecoff[1]/np.sum(exspecoff[1])/interprefspec(exspecoff[0]),label='Dark')

# plt.legend()
# exspec=rpl.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_mid*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
# plt.plot(exspec[0],(exspec[1]/np.sum(exspec[1])))
plt.xlabel('Excitation wavelength')
plt.ylabel('Intensity')
plt.title('Excitation spectra')




# plt.figure()
# emdata=Yfiltered[:,beginbin:endbin]
# # plt.plot(excitationdata[:,1],np.sum(excitationdata,axis=1)/np.sum(exspecex)/(interprefspec(excitationdata1[1][:,1])),label='Bright')
# plt.plot(wavelengths,np.sum(emdata,axis=1)/np.sum(emdata),label='Bright')
# # plt.plot(wavelengths,np.sum(emspectr,axis=1)/np.sum(emspectr),label='Gray')
# # plt.plot(wavelengths,np.sum(emspecoff,axis=1)/interprefspec(wavelengths)/np.sum(emspecoff),label='Dark')
# # plt.plot(excitationdata1[1][:,1],np.sum(exspecoff,axis=1)/np.sum(exspecoff)/interprefspec(excitationdata1[1][:,1]),label='Dark')
# plt.title('Emission spectra')
# plt.ylabel('Intensity')
# plt.xlabel('Emission wavelength (nm)')
# plt.xlim(590,620)
# # plt.legend(loc=0)
#%% Limits
limitex=2000 #base these limits on the above-mentioned plot.
limittrlow=800
limittrhigh=2000


limitoffhigh=600
# 
lifetimeexciton=50
beginbin=100
endbin=150
# lifetimetrionmin=4
# lifetimetrionmax=6
# microtimesin=microtimesblue
# macrotimesin=macrotimesblue
macrocyclelist=data[19]
threshlow=1/Freq/4
threshhigh=3/Freq/4
Z = np.logical_and(threshlow<(macrocyclelist*dtmacro),(macrocyclelist*dtmacro)<= threshhigh)
tforward=macrocyclelist[np.where(Z)]
tbackward=macrocyclelist[np.where(np.logical_not(Z))]
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
idxextr=np.zeros(len(limits))-1
# for tbinnr in range(len(limits)-1):
for tbinnr in range(beginbin,endbin-1):
    nphots = limits[tbinnr+1]-limits[tbinnr]
    microtimes = microtimesin[limits[tbinnr]:limits[tbinnr+1]]
    macrotimescycle = macrotimescyclein[limits[tbinnr]:limits[tbinnr+1]]

    if nphots>limitex:# and taulist[tbinnr]<lifetimeexciton:

        microtimes_ex[nrex:nrex+nphots]=microtimes
        macrotimescycle_ex[nrex:nrex+nphots]=macrotimescycle
        bins_ex+=1
        nrex+=nphots
        timeexciton[tbinnr]=1
        idxextr[tbinnr]=0


    elif nphots>limittrlow and nphots<limittrhigh and taulist[tbinnr]<10 and tauav[tbinnr]>1:# and taulist[tbinnr]>0.05: #and (photonspbin[tbinnr]-7)/taulist[tbinnr]>112/28
        # print('found')

#    elif photonspbin[tbinnr]>limittrlow and photonspbin[tbinnr]<limittrhigh and taulist[tbinnr]>0.5 and (photonspbin[tbinnr]-10)/taulist[tbinnr]>505/27:
        microtimes_trion[nrtrion:nrtrion+nphots]=microtimes
        macrotimescycle_trion[nrtrion:nrtrion+nphots]=macrotimescycle
        bins_trion+=1
        nrtrion+=nphots
        timetrion[tbinnr]=1
        idxextr[tbinnr]=1
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
fitex=rpl.GetLifetime(microtimes_ex,dtmicro,dtmacro,200e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_ex/(Texp/binwidth),expterms=1,method='ML_c')
# fitex=GetLifetime(microtimes_ex,dtmicro,dtmacro,100e-9,tstart=-1,plotbool=True,ybg=bins_ex*40*binwidth/np.max(microtimesblue),method='ML_c')
fittrion=rpl.GetLifetime(microtimes_trion,dtmicro,dtmacro,5e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_trion/(Texp/binwidth),expterms=1,method='ML_c')
# fittrion=GetLifetime(microtimes_trion,dtmicro,dtmacro,5e-9,tstart=-1,plotbool=True,ybg=bins_trion*40*binwidth/np.max(microtimesblue),method='ML_c')

# fitoff=rpl.GetLifetime(microtimes_off,dtmicro,dtmacro,10e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_trion/(Texp/binwidth),method='ML_c')
# fitmid=rpl.GetLifetime(microtimes_off,dtmicro,dtmacro,200e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_mid/(Texp/binwidth),expterms=1,method='ML_c')

# plt.figure()

# plt.figure()
# plt.semilogy(fitex[4],fitex[5])
# plt.semilogy(fittrion[4],fittrion[5])
# plt.semilogy(fitmid[4],fitmid[5])
# plt.title('Decay histograms (sum)')
# plt.xlabel('time (ns)')
# plt.ylabel('counts (a.u.)')

# # print('Rad lifetime ratio:'+str(fittrion[1]/bins_trion/(fitex[1]/bins_ex)))

# plt.figure()
# plt.semilogy(fitex[4],fitex[5]/bins_ex)
# plt.semilogy(fittrion[4],fittrion[5]/bins_trion)
# plt.semilogy(fitmid[4],fitmid[5]/bins_off)
# plt.title('Decay histograms (avg)')
# plt.xlabel('time (ns)')
# plt.ylabel('counts (a.u.)')

def pileupcorrection(data,dtmacro,lifetime):
    corrected = data*(1-np.e**(-dtmacro/lifetime))
    return corrected

plcorrex=pileupcorrection(fitex[5],dtmacro,fitex[0]/1e9)
plcorrtrion=pileupcorrection(fittrion[5],dtmacro,fittrion[0]/1e9)
plcorroff=pileupcorrection(fitmid[5],dtmacro,0.5/1e9)

plt.figure()
plt.semilogy(fitex[4],plcorrex/bins_ex, label='Bright')
plt.semilogy(fittrion[4],plcorrtrion/bins_trion, label='Gray')
# plt.semilogy(fitmid[4],plcorroff/bins_off,label='Dark')
plt.legend()
plt.title('Decay histograms')#' (avg+pile-up corr)')
plt.xlabel('time (ns)')
plt.ylabel('counts (a.u.)')


print('Rad lifetime ratio:'+str(fittrion[1]/bins_trion/(fitex[1]/bins_ex)))
#plt.xlim([0,220])
# plt.legend(['High cps','High cps fit','Mid cps','Mid cps fit','Low cps','Low cps fit'])

#%%

histsettings=(505,575,100)
plt.figure()
exspecex=rplm.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_ex*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
plt.plot(exspecex[0],exspecex[1]/np.sum(exspecex[1])/interprefspec(exspecex[0]),label='Bright')
exspectr=rplm.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_trion*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
plt.plot(exspectr[0],exspectr[1]/np.sum(exspectr[1])/interprefspec(exspectr[0]),label='Gray')
exspecoff=rplm.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_off*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
# plt.plot(exspecoff[0],exspecoff[1]/np.sum(exspecoff[1])/interprefspec(exspecoff[0]),label='Dark')

plt.legend()
# exspec=rpl.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_mid*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
# plt.plot(exspec[0],(exspec[1]/np.sum(exspec[1])))
plt.xlabel('Excitation wavelength')
plt.ylabel('Intensity')
plt.title('Excitation spectra')

#%% emission
emspecex=np.zeros(Yfiltered.shape)
emspectr=np.zeros(Yfiltered.shape)
emspecoff=np.zeros(Yfiltered.shape)

exspecex=np.zeros(excitationdata.shape)
exspectr=np.zeros(excitationdata.shape)
exspecoff=np.zeros(excitationdata.shape)

nrex=0
nrtr=0
nroff=0

tempsum=np.sum(Yfiltered,axis=0)
for j in range(len(Yfiltered[0])):
    # if tempsum[j]>intensitymaxoff:
    if idxextr[j]==0:
        nrex+=1
        emspecex[:,j]=Yfiltered[:,j]
        exspecex[:,j]=excitationdata[:,j]
    elif idxextr[j]==1:
        nrtr+=1
        emspectr[:,j]=Yfiltered[:,j]
        exspectr[:,j]=excitationdata[:,j]
    # else:
    #     nroff+=1
    #     emspecoff[:,j]=Yfiltered[:,j]
    #     exspecoff[:,j]=excitationdata[:,j]
        
plt.figure()
plt.plot(wavelengths,np.sum(emspecex,axis=1)/np.sum(emspecex),label='Bright')        
plt.plot(wavelengths,np.sum(emspectr,axis=1)/np.sum(emspectr),label='Gray')
# plt.plot(wavelengths,np.sum(emspecoff,axis=1)/np.sum(emspecoff),label='Dark')
plt.title('Emission spectra')
plt.ylabel('Intensity')
plt.xlabel('Emission wavelength (nm)')
plt.legend()
plt.xlim(590,620)

#%%
histsettings=(430,580,100)
plt.figure()
exspecex=rplm.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_ex*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
plt.plot(exspecex[0],exspecex[1]/bins_ex/interprefspec(exspecex[0]),label='Bright')
exspectr=rplm.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_trion*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
plt.plot(exspectr[0],exspectr[1]/bins_trion/interprefspec(exspectr[0]),label='Gray')
# exspecoff=rplm.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_off*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
# plt.plot(exspecoff[0],exspecoff[1]/np.sum(exspecoff[1])/interprefspec(exspecoff[0]),label='Dark')

plt.legend()
# exspec=rpl.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_mid*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
# plt.plot(exspec[0],(exspec[1]/np.sum(exspec[1])))
plt.xlabel('Excitation wavelength')
plt.ylabel('Intensity')
plt.title('Excitation spectra')
#%% 2 detectors
macrotimesindet0=data[3]
macrotimesindet1=data[5]
microtimesindet0=data[2]
microtimesindet1=data[4]
binwidth=0.01
macrolimitsdet0 = rpl.HistPhotons(macrotimesindet0*dtmacro,binwidth,Texp)
macrolimitsdet1 = rpl.HistPhotons(macrotimesindet1*dtmacro,binwidth,Texp)
lengthlimits=int(Texp/binwidth)
macrotimesin=data[25]
macrolimitsdet0=macrotimesin
macrolimitsdet1=macrotimesin
lengthlimits=len(data[25])
limitsdet0=macrolimitsdet0
limitsdet1=macrolimitsdet1
taulist=np.zeros(lengthlimits-1)

taulist1=np.zeros(lengthlimits-1)
taulist2=np.zeros(lengthlimits-1)

#wavelbins=150
#Exspeclist=np.zeros([wavelbins,lengthlimits-1])
tauavdet0=np.zeros(lengthlimits-1)
tauavdet1=np.zeros(lengthlimits-1)
tauavtot=np.zeros(lengthlimits-1)
Alist=np.zeros(lengthlimits-1)
Alist1=np.zeros(lengthlimits-1)
Alist2=np.zeros(lengthlimits-1)
photonspbindet0=np.zeros(lengthlimits-1)
photonspbindet1=np.zeros(lengthlimits-1)
# photonspbintot=np.zeros(lengthlimits-1)
ybglist=np.zeros(lengthlimits-1)
buff=np.zeros(lengthlimits-1)
meanex=np.zeros(lengthlimits-1)
stdex=np.zeros(lengthlimits-1)
histbinmultiplier=1
plt.figure()
lifetime1=rpl.GetLifetime(data[4],dtmicro,dtmacro,250e-9,tstart=-1,histbinmultiplier=1,ybg=0,plotbool=True,expterms=2,method='ML_c')
plt.show()
#[taulist,Alist,ybglist] = Parallel(n_jobs=-1, max_nbytes=None)(delayed(processInput)(tbinnr) for tbinnr in tqdm(range(nrtbins-1)))
#plt.figure()
#test=np.zeros((wavelbins,lengthlimits-1))
begintime=0
endtime=100
beginidx=int(begintime/(data[3][-1]*dtmacro)*lengthlimits)
endidx=int(endtime/(data[3][-1]*dtmacro)*lengthlimits)
# for tbinnr in tqdm(range(beginidx,endidx-1)):
for tbinnr in tqdm(range(lengthlimits-1)):
    microtimesdet0 = microtimesindet0[limitsdet0[tbinnr]:limitsdet0[tbinnr+1]]
    microtimesdet1 = microtimesindet1[limitsdet1[tbinnr]:limitsdet1[tbinnr+1]]

    # [taulist[tbinnr],Alist[tbinnr],ybglist[tbinnr],buff[tbinnr]]=rpl.GetLifetime(microtimes,dtmicro,dtmacro,300e-9,tstart=lifetime[3]*dtmicro,histbinmultiplier=1,ybg=lifetime[2]*binwidth/Texp,plotbool=False,expterms=1,method='ML_c') 
    # lifetimetot=rpl.GetLifetime(microtimes,dtmicro,dtmacro,300e-9,tstart=lifetime[3]*dtmicro,histbinmultiplier=1,ybg=lifetime[2]*binwidth/Texp,plotbool=False,expterms=2,method='ML_c') 
    # [taulist1[tbinnr],taulist2[tbinnr],Alist1[tbinnr],Alist2[tbinnr],ybglist[tbinnr],buff[tbinnr]]=[lifetimetot[0][0],lifetimetot[0][1],lifetimetot[1][0],lifetimetot[1][1],lifetimetot[2],lifetimetot[3]]
    tauavdet0[tbinnr]=(np.mean(microtimesdet0)-lifetime[3])*dtmicro*1e9-1
    tauavdet1[tbinnr]=(np.mean(microtimesdet1)-lifetime[3])*dtmicro*1e9-1
    tauavtot[tbinnr]=np.mean([tauavdet0[tbinnr],tauavdet1[tbinnr]]) #not so sure about this line actually
    photonspbindet0[tbinnr]=len(microtimesdet0)
    photonspbindet1[tbinnr]=len(microtimesdet1)
photonspbintot=photonspbindet0+photonspbindet1
    #using th
    #for when histogramming photons
if lengthlimits==len(data[25]):
    #for when correlating emission    
    fig,ax1 = plt.subplots()
    ax1.plot(data[26][:-1]*dtmacro,photonspbintot,'b')
    #ax1.plot(macrotimesin[limits[0:lengthlimits-1]]*dtmacro,Alist,'b')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Photons per emission bin', color='b')
    #ax1.set_xlim([0,13])
    #ax1.set_ylim([0,0.2*np.max(ylistspec)])
    #ax1.ticklabel_format(style='sci',scilimits=(-2,3),axis='both')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(data[26][:-1]*dtmacro,tauavtot,'r')
    #ax2.set_ylim([586,588])
    ax2.set_ylim([0,50])
    ax2.set_ylabel('Lifetime (ns)', color='r')
    ax2.tick_params('y', colors='r')
    #ax2.ticklabel_format(style='sci',scilimits=(-2,3),axis='both')

else:
    fig,ax1 = plt.subplots()
    ax1.plot(macrotimesindet0[macrolimitsdet0[0:lengthlimits-1]]*dtmacro,photonspbintot,'b')
    #ax1.plot(macrotimesin[limits[0:lengthlimits-1]]*dtmacro,Alist,'b')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Photons per '+str(int(binwidth*1000))+' ms bin', color='b')
    #ax1.set_xlim([0,13])
    #ax1.set_ylim([0,0.2*np.max(ylistspec)])
    #ax1.ticklabel_format(style='sci',scilimits=(-2,3),axis='both')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(macrotimesindet0[macrolimitsdet0[0:lengthlimits-1]]*dtmacro,tauavdet1,'r')
    #ax2.set_ylim([586,588])
    ax2.set_ylim([0,50])
    ax2.set_ylabel('Lifetime (ns)', color='r')
    ax2.tick_params('y', colors='r')





#%% Plot FLID map
plt.figure()
# photonspbintot=photonspbintot[photonspbintot>1/1000]
# tauavtot=tauavtot[photonspbintot>1/1000]
# photonspbintot=np.delete(photonspbintot,np.where(photonspbintot.ravel()==0))
# tauavtot=np.delete(tauavtot,np.where(tauavtot.ravel()==0))
# plt.hist2d(tauav,photonspbin,bins=(50,int(np.max(photonspbin))),range=[[0,50],[0,np.max(photonspbin)]],norm=mcolors.LogNorm())
plt.hist2d(tauavtot,photonspbintot,bins=(50,int(np.max(photonspbintot))),range=[[0,50],[0,np.max(photonspbintot)]],norm=mcolors.LogNorm())
# plt.hist2d(taulist2,photonspbin,(50,int(np.max(photonspbin))),range=[[0,50],[0,np.max(photonspbin)]],norm=mcolors.LogNorm())
plt.title('FLID map')
plt.ylabel('Counts per bin')
plt.ylabel('Counts per '+str(int(binwidth*1000))+' ms bin')
plt.xlabel('Lifetime (ns)')
plt.colorbar()

#plt.plot(np.arange(0,50),np.arange(0,50)*505/27+8,'w')


#%% fit absorption curve
#at the moment it does not work and I do not know why. Based it a bit on the cubic background david wrote in his paper, but i can imagine that it is highly dependent on the starting positions



# plt.figure()
# plt.plot(exspecex[0],exspecex[1]/interprefspec(exspecex[0]))
# plt.figure()
# plt.plot(excspecwav[0],excspeccorr[0]) 
indices=(430,562)
wavelindices=(np.argmin(np.abs(excspecwav[0]-indices[0])),np.argmin(np.abs(excspecwav[0]-indices[1])))

spectra=excspeccorr[0][wavelindices[0]:wavelindices[1]]
testwav=excspecwav[0][wavelindices[0]:wavelindices[1]]

def func(x, a, b, c, d):
    return a + b * x + c * x ** 2 + d * x ** 3

lormod = LorentzianModel(prefix='Lor_')
pars = lormod.guess(spectra, x=testwav)
pars['Lor_center'].set(value=440)
lormod2=LorentzianModel(prefix='Lor2_')
pars.update(lormod2.make_params())
pars['Lor2_center'].set(value=480)
lormod3=LorentzianModel(prefix='Lor3_')
pars.update(lormod3.make_params())
pars['Lor3_center'].set(value=520)
lormod4=LorentzianModel(prefix='Lor4_')
pars.update(lormod4.make_params())
pars['Lor4_center'].set(value=540)
# pmodel = Model(func)
# pars.update(pmodel.make_params())
# linmodel=LinearModel(prefix='lin_')
# pars.uodate(linmodel.make_params())

# bkg = ConstantModel(prefix='contst')
# pars.update(bkg.make_params())
bkg2 = LinearModel(prefix='lin')
pars.update(bkg2.make_params())

mod=lormod+lormod2+lormod3+lormod4+bkg2
init = mod.eval(pars, x=testwav)
out = mod.fit(spectra, pars, x=testwav)
# exspecex

plt.figure()
plt.plot(testwav,out.best_fit)
plt.plot(testwav,spectra)


#%% Lifetime correlation
# this idea is based on not using the GetLifeTime function to generate lifetimes, but instead make a 2D correlation map with the observed microtime (and macrotime), separated by a time interval of dT


shortset = 1000 #not analyzing all the data to save time during checks
microtimes = (data[2][0:shortset]-lifetime[3]) #nanoseconds
macrotimes = data[3][0:shortset]#seconds
# dt = 0.05 #seconds
# dt = microtimes[1]-microtimes[0] #seconds
# timerange = 1e-1
# dt = macrotimes[1]-macrotimes[0] #seconds
# timerange = 1e-4


# krange = int(shortset/100)
krange = shortset
macrotimesindex = np.zeros((len(macrotimes),krange))

tmin=0
tmax=1/100


microtimes1 = np.zeros(len(macrotimes)**2)
microtimes2 = np.zeros(len(macrotimes)**2)
# microtimes2 = np.zeros((len(macrotimes),krange))
# microtimes3 = np.zeros((len(macrotimes),krange))

pairsfound=0
for j in tqdm(range(shortset)):
    for k in range(shortset): #krange actually
        if tmin<=np.abs(macrotimes[j]-macrotimes[k])<=tmax:
            microtimes1[pairsfound]=microtimes[j]
            microtimes2[pairsfound]=microtimes[k]
            # microtimes1[j]=microtimes[j]
            # microtimes2[j][k]=microtimes[k]
            pairsfound+=1
            # microtimes3[j][k]=microtimes[j]
            # microtimes2[j][k]=k
            # microtimes3[j][k]=j
            # microtimes1[j]=j
            # microtimes2[j][k]=k
 
microtimes1 = microtimes1[0:pairsfound]
microtimes2 = microtimes2[0:pairsfound]           


hist,xedge,yedge = np.histogram2d(microtimes1,microtimes1,bins=100)
    

plt.figure()
plt.hist2d(microtimes1,microtimes2,bins=100)

# plt.figure()
# plt.imshow(hist)       
# plt.colorbar()

           #now thingss get ugly in the sense that you now have to do this laplace stuff

 


#%% Close all open figures in all scripts
plt.close('all')

#%% savefig


plt.savefig('E:/Martijn/ETH/results/20200310_PM111_specdiffusion/QD2/Correlation_map_tau0',dpi=800)
