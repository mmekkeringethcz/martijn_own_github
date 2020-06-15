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
import socket #enables to find computer name to make less of a mess with folders
#%matplotlib auto
sys.path.append('E:/Martijn/ETH/scripts/github/martijn_own_github/repoo/apd_ple')
import Read_PLE_functions as rpl
sys.path.append('E:/Martijn/ETH/scripts/github/martijn_own_github/repoo/martijn_own_github')
import Read_PLE_functions_Martijn as rplm
InVoltagenew_c=nb.jit(nopython=True)(rplm.InVoltagenew)

#%%

def histogramexcitationspectra(rawdata,originaldata,wavelengthrange,histogram):
#rawdata is the firstphoton of each cycle of interest.Original data is used to not cover more photons when the emission wavelength does not match.
    binnedintensities = np.zeros((histogram,len(rawdata)))
    binnedwavelengths = np.zeros((histogram,len(rawdata)))
    
    for i in range(len(rawdata)-1):
        originaldataphotonlistindex = np.argmin(np.abs(originaldata-rawdata[i]))
        [intensitiestemp,wavelengthstemp] = np.histogram(InVoltagenew_c(dtmacro*data[19][rawdata[i]:originaldata[originaldataphotonlistindex+1]],Freq,Vpp,Voffset,tshift)*calibcoeffs[0]+calibcoeffs[1],histogram,range=wavelengthrange)
        binnedintensities[:,i]=intensitiestemp
        wavelengthstempavg = (wavelengthstemp[:-1]+0.5*(wavelengthstemp[1]-wavelengthstemp[0]))
        binnedwavelengths[:,i]= wavelengthstempavg
        
    return binnedintensities,binnedwavelengths
# In[25]:


### GENERAL EVALUATION OF TTTR data     

# parameters (use forward slashes!)
Debugmode=False     #Set to true to make plots in the individual sections of the code to check that everything is working properly

basefolders={'DESKTOP-BK4HAII':'C:/Users/rober/Documents/Doktorat/Projects/SingleParticle_PLE/Andor Spectrometer/',
             'HP_Probook':'E:/Martijn/ETH/results/',
             'mavt-omel-w004w':'E:/LAB_DATA/Robert/'} #dictionary which base folder to use on which computer. For you this probably would be everything until 
folder = basefolders[socket.gethostname()]+'20200612_HR148_RT/'

filename = 'HR148_9p734MHz_200Hz_500mVpp_60mVoffset_QD2_ND1'

settingsfile= filename+'_settings'
HHsettings=rpl.load_obj(settingsfile, folder )
# HHsettings={}

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
lifetime=rpl.GetLifetime(microtimes0,dtmicro,dtmacro,dtfit=400e-9,tstart=-1,histbinmultiplier=1,ybg=-1,plotbool=True,method='ML_c')

limits0 = rpl.HistPhotons(times0*dtmicro,binwidth,Texp) #gives index of first photon of det0 in each bin
limits1 = rpl.HistPhotons(times1*dtmicro,binwidth,Texp)

# make an intensity trace and find Imax
inttrace = rpl.MakeIntTrace(limits0,limits1,binwidth,Texp)
#%% Wavelength calibration
Voltage=np.array([-80,-40,0,40,80])
Wavelength_calib=np.array([578.6,560.2,541.5,560.3,504.9])
calibcoeffs=np.polyfit(Voltage,Wavelength_calib,1)
Vpp=float(filename.split('mVpp_')[0].split('_')[-1])
Freq=float(filename.split('Hz_')[-2])
Voffset=float(filename.split('mVoffset')[0].split('Vpp_')[-1])
# Voffset=0
tshift=rplm.Findtshift(Freq,Vpp,Voffset,calibcoeffs,data[19],dtmacro,(500,570),(-6e-4,-2e-4)) #coarse sweep
tshift=rplm.Findtshift(Freq,Vpp,Voffset,calibcoeffs,data[19],dtmacro,(500,570),(tshift-1e-5,tshift+1e-5),Debugmode=True)

    


#%% laser intensity corrected
    #done by sweeping laser spectrum over camera and divide QD em spectrum over normalized reference spectrum
    
reffolder = folder
calibspec = rpl.importASC(reffolder+'backref_9p734MHz_200Hz_500mVpp_60mVoffset.asc')

# plt.figure()
# plt.imshow(calibspec[0],extent=[np.min(calibspec[1]),np.max(calibspec[1]),np.max(calibspec[1]),np.min(calibspec[1])])
# plt.gca().invert_yaxis()
# plt.figure()
# plt.imshow(refspec[0])
cmaxpos=np.argmax(np.mean(calibspec[0][:,:],1))
refspecold=np.mean(calibspec[0][cmaxpos-5:cmaxpos+5,:],0)
refspec=np.mean(calibspec[0][cmaxpos-5:cmaxpos+5,:],0)#-np.mean(refspecold[480:len(refspecold)])
refspec=refspec*1.2398/calibspec[1]
refspec=refspec/np.max(refspec)
# refspec=savgol_filter(refspec, 23, 3)
interprefspec= interp1d(calibspec[1], refspec,kind='cubic',fill_value='extrapolate')
# plt.figure()
# # plt.plot(np.linspace(500,600,300),interprefspec(np.linspace(500,600,300)))
# plt.plot(calibspec[1],refspec)



# refspecwav = interprefspec(np.linspace(np.min(tlistforward),np.max(tlistforward),num=histbinnumber))

# if Debugmode==True:
#     plt.figure()
#     plt.plot(calibspec[1],interprefspec(calibspec[1]))
#     # plt.plot(tlistbackward,ylistbackward/interprefspec(calibspec[1]))
    
#     plt.figure()
#     plt.plot(tlistforward,ylistforward/interprefspec(calibspec[1]))
#     plt.plot(tlistbackward,ylistbackward/interprefspec(calibspec[1]))
    
#     plt.figure()
#     plt.plot(Wavelengthspec,interprefspec(ylistspec))
# #check whether this excitation regime actually extends in the excitation spectrum of the regime you are interested in.
wavelengthrangemin,wavelengthrangemax = 500,593
wavelengthrange = (wavelengthrangemin,wavelengthrangemax)
histogrambin=247
[excspec,excspecwavtemp] = np.histogram(InVoltagenew_c(dtmacro*data[19],Freq,Vpp,Voffset,tshift)*calibcoeffs[0]+calibcoeffs[1],histogrambin,range=wavelengthrange)
excspecwav = [excspecwavtemp[:-1]+(excspecwavtemp[1]-excspecwavtemp[0])/2]
excspeccorr = excspec/interprefspec(excspecwav)
plt.figure()
plt.plot(excspecwav[0],excspeccorr[0])   
plt.title('Laser corrected') 

plt.figure()
plt.plot(excspecwav[0],excspec)    
plt.title('Original') 


#%% Generate binned excitation spectra
#this generates part of unbinned excitation spectra. It does not resolve for forward and backward scan
binning = 1/10 #binning is in units of seconds (s)
expplottime = data[17][-1]*dtmacro
excitationbegin = data[17][0]*dtmacro
# binning = 1 #binning is in units of seconds (s)
wavelengthrangemin,wavelengthrangemax = 500,593
wavelengthrange = (wavelengthrangemin,wavelengthrangemax)
histogrambin=247 #number is irrelevant for excitation and emission correlation. Fixed that.
# histogrambin= int(np.divide((wavelengthrangemax-wavelengthrangemin),(np.max(Wavelengthspec)-np.min(Wavelengthspec)))*1/dtmacro/200/40) #200 is frequency of glvo cycle
# firstphotonlist = rpl.HistPhotons(times0*dtmicro,binning,Texp)
firstphotonlist = rpl.HistPhotons(times0*dtmicro,binning,expplottime)
binnedintensities = np.zeros((histogrambin,len(firstphotonlist)))
binnedwavelengths = np.zeros((histogrambin,len(firstphotonlist)))



for i in range(len(firstphotonlist)-1):
    [intensitiestemp,wavelengthstemp] = np.histogram(InVoltagenew_c(dtmacro*data[19][firstphotonlist[i]:firstphotonlist[i+1]],Freq,Vpp,Voffset,tshift)*calibcoeffs[0]+calibcoeffs[1],histogrambin,range=wavelengthrange)
    wavelengthstempavg = (wavelengthstemp[:-1]+0.5*(wavelengthstemp[1]-wavelengthstemp[0]))
    binnedwavelengths[:,i]= wavelengthstempavg
    binnedintensities[:,i]=intensitiestemp#/interprefspec(wavelengthstempavg)
    

if Debugmode==True:
    plt.figure()
    plt.plot(binnedwavelengths,binnedintensities)
    plt.xlim(np.min(np.delete(binnedwavelengths.ravel(),np.where(binnedwavelengths.ravel()<=1))),np.max(binnedwavelengths))
    plt.xlabel('Excitation Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('binning = ' +str(binning) +'s')
    
    
if Debugmode==True:    
    plt.figure()
    plt.imshow(binnedintensities,extent=[excitationbegin,expplottime,np.max(binnedwavelengths[:,1]),np.min(binnedwavelengths[:,1])])
    plt.gca().invert_yaxis()
    plt.ylabel('Excitation wavelength (nm)')
    plt.xlabel('time (s)')
    plt.title('binning = ' +str(binning))
    plt.colorbar()
    
binnedintensitiesmean = np.mean(binnedintensities,axis=1) #timeaverage
binnedintensitiesmean2 = np.mean(binnedintensities,axis=0) #wavelengthaverage
binnedintensitiesmeancorr = np.zeros(binnedintensities.shape)
binnedintensitiesmeancorr2 = np.zeros(binnedintensities.shape)

for j in range(len(binnedintensities[0])):
    # binnedintensitiesmeancorr[j,:] = binnedintensities[j,:] - binnedintensitiesmean[j]
    binnedintensitiesmeancorr2[:,j] = binnedintensities[:,j] - binnedintensitiesmean2[j]
    
if Debugmode==True:
    vmin = 0
    vmax=10
    vmax=np.max(binnedintensitiesmeancorr2)
    plt.figure()
    plt.imshow(binnedintensitiesmeancorr2,extent=[excitationbegin,expplottime,np.max(binnedwavelengths[:,1]),np.min(binnedwavelengths[:,1])],vmin=vmin,vmax=vmax)
    plt.gca().invert_yaxis()
    plt.ylabel('Excitation wavelength (nm)')
    plt.xlabel('time (s)')
    plt.title('binning = ' +str(binning))
    # plt.ylim(0,10)
    plt.colorbar()
    
    plt.figure()
    plt.plot(binnedwavelengths,binnedintensitiesmean)
    plt.xlim(np.min(np.delete(binnedwavelengths.ravel(),np.where(binnedwavelengths.ravel()<=1))),np.max(binnedwavelengths))
    plt.xlabel('Excitation Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('binning = ' +str(binning))
    
    plt.figure()
    plt.plot(np.arange(data[17][0]*dtmacro,data[17][-1]*dtmacro,len(binnedintensities[0])),np.sum(binnedintensities,axis=0))
    
# plt.figure()
# plt.plot(binnedwavelengths,binnedintensities)
# plt.xlim(np.min(np.delete(binnedwavelengths.ravel(),np.where(binnedwavelengths.ravel()<=1))),np.max(binnedwavelengths))
# plt.xlabel('Excitation Wavelength (nm)')
# plt.ylabel('Intensity')
# plt.title('binning = ' +str(binning))
    
#%% excitation correlation

taus=[0,1,2,10,20]
plot = 'corr'
plot = 'cov'

Excitationdata1 = binnedintensities
Excitationdata2 = binnedintensities


excitationcovariance,excitationnormalization,excitationcorrelation = rplm.pearsoncorrelation(Excitationdata1,Excitationdata2,binnedwavelengths[:,1],binnedwavelengths[:,1],taus=taus,plot=plot)


#%%focus on the 575 peak

plot = 'corr'
taus = [0,1,10,100]
taus = [0,1,2,5,10,20]
# taus = np.arange(10)
# wavmin = 140
wavmin = 1
binnedwavelengths[wavmin]

# wavmax = 180
wavmax = len(binnedwavelengths)
binnedwavelengths[wavmax-1]

timemin = 0
timemax = 1100

excitationdata1 = binnedintensities[wavmin:wavmax+1,timemin:timemax]
excitationdata2 = binnedintensities[wavmin:wavmax+1,timemin:timemax]


excitationcovariance,excitationnormalization,excitationcorrelation = rplm.pearsoncorrelation(excitationdata1,excitationdata2,binnedwavelengths[:,1][wavmin:wavmax+1],binnedwavelengths[:,1][wavmin:wavmax+1],taus=taus,plot=plot)
# excitationcovariance,excitationnormalization,excitationcorrelation = spectralcorrelation(excitationdata1,excitationdata2,binnedwavelengths[:,1][wavmin:wavmax+1],binnedwavelengths[:,1][wavmin:wavmax+1],taus=taus,plot=plot)
#%% Read emission spectra
    
emissionspec = rpl.importASC(folder+filename+'.asc')
# emissionspec = rpl.importASC(folder+'Kineticseries_LED_Fluocube_midPower_noND.asc')
#produces array of emission spectra 
wavelengthsinit = emissionspec[1]
numberypixels = len(emissionspec[0])
numberspectra = len(emissionspec[0][0][0])


#%%
if len(emissionspec[0])==1:
    Ybackground=475
    Yfiltered = emissionspec[0][0]-Ybackground
    wavelengths = wavelengthsinit
    numberypixels=1
    timeaverage = np.mean(emissionspec[0],axis=2) 
    
    if Debugmode==True:
        plt.figure()
        plt.plot(wavelengths,timeaverage.ravel())
        plt.xlabel('Emission wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title('time-averaged signal')
        
        emwavelrange=(595,630)
        emwavelindices=(np.argmin(np.abs(wavelengths-emwavelrange[0])),np.argmin(np.abs(wavelengths-emwavelrange[1])))
        timeminplot=0
        timemaxplot=4000
        plt.figure()
        plt.imshow(Yfiltered[emwavelindices[0]:emwavelindices[1],timeminplot:timemaxplot],extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths[emwavelindices[0]:emwavelindices[1]]),np.min(wavelengths[emwavelindices[0]:emwavelindices[1]])],aspect='auto')#,vmax=1000)
        # plt.imshow(Yfiltered,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.min(wavelengths),np.max(wavelengths)],aspect='auto')
        # plt.imshow(Yfiltered)
        plt.ylabel('Emission wavelength (nm)')
        plt.xlabel('Time (s)')
        plt.title('intensity vs time per wavelength')
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
    
    nrspec = 350 #first so many emission wavelengths that are neglected
    nrfinspec = -1 #last so many emission wavelengths that are neglected
    
    wavelengths = wavelengthsinit[nrspec:nrfinspec]
    Yfilteredwithbkg = Yfilteredinit[nrspec:nrfinspec]
    timeaveragewithbkg = timeaverageinit[nrspec:nrfinspec]
    
    Yfiltered = Yfilteredwithbkg-Ybackground #Yfilterd is now background corrrected.
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




#%% Spectral correlation of noise
        #done by selecting window where the QD does no emit. Hope to see that everything is as zero as it can be
        #maps look really weird
test = np.mean(emissionspec[0][0:30],axis=0)
test=test[0:-1,:]
taus=[0,1,2]


Emissiondata1 = test
Emissiondata2 = test

emissioncovariance,emissionnormalization,emissioncorrelation = rplm.pearsoncorrelation(Emissiondata1,Emissiondata2,wavelengths,wavelengths,taus=taus,plot=plot)

#%% Spectral correlation of emission using covariances and pair counting
    # up til now only postselection


# taus = np.arange(0,50,1)
# taus=[0,5,10,100]
taus=[0,1,2]

taus=[1]
# taus=[1,2,10,15]
# plot = 'corr'
plot='norm'
plot='all'
# plot = 'cov'

# wavmin = 80
# wavelengths[wavmin]

# wavmax = 300
# wavelengths[wavmax]

timemin = 0
timemax = 2400
Emissiondata1 = Yfiltered[:,timemin:timemax] #can get postselection by selecting only a particular window. Note that you need emissiondata to plot insteada of Yfiltered
Emissiondata2 = Yfiltered[:,timemin:timemax]

# Emissiondata1 = Yfiltered[wavmin:wavmax,15:23] #can get postselection by selecting only a particular window. Note that you need emissiondata to plot insteada of Yfiltered
# Emissiondata2 = Yfiltered[wavmin:wavmax,15:23]

# Emissiondata1 = Yfiltered[:,15:23] #can get postselection by selecting only a particular window. Note that you need emissiondata to plot insteada of Yfiltered
# Emissiondata2 = Yfiltered[:,15:23]
# emissioncovariance,emissionnormalization,emissioncorrelation = rplm.pearsoncorrelation(Yfiltered,Yfiltered,wavelengths,wavelengths,taus=taus,plot=plot)
# emissioncovariance,emissionnormalization,emissioncorrelation = spectralcorrelation(Yfiltered,Yfiltered,wavelengths,wavelengths,taus=taus,plot=plot)

emissioncovariance,emissionnormalization,emissioncorrelation = rplm.pearsoncorrelation(Emissiondata1,Emissiondata2,wavelengths,wavelengths,taus=taus,plot=plot)

#%% Generate binned excitation spectra for excitation emission correlation
# #select particular exctiation wavelength based on chosen range emission wavelengths


rawdata = data[25]
originaldata= data[25]
wavelengthrange = (500,593)
histogram = 200
excitationdata = histogramexcitationspectra(rawdata,originaldata,wavelengthrange,histogram)

for j in range(len(excitationdata[0][0])):
    excitationdata[0][:,j] = excitationdata[0][:,j]-np.mean(excitationdata[0][:,j])
    
plt.figure()
plt.imshow(excitationdata[0],extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,wavelengthrange[1],wavelengthrange[0]],vmin=0)
plt.gca().invert_yaxis()
plt.colorbar()
plt.ylabel('Excitation Wavelength (nm)')
plt.xlabel('time (s)')
plt.title('Full excitation spectra')  

#%% #select particular exctiation wavelength based on chosen range emission wavelengths
emwavelrange = np.array([(606.5,609.5),(611.5,614.5),(615,618)])
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
#%%
# for j in range(len(emwavelrange)):
idx=0
photonlist = np.argwhere(maxindex==idx).ravel()
rawdata = data[25][photonlist]
lengthdataset= len(photonlist) #used for plotting on scale
times= data[26][photonlist]*dtmacro

wavelengthrange = (500,593)
histogram = 200
excitationdata = rplm.histogramexcitationspectra(rawdata,data[25],wavelengthrange,histogram)

plotexcitationdata=np.zeros((histogram,len(data[25]))) #done to match the indexes
idxs=np.zeros(len(photonlist))
for j in range(len(photonlist)):
    idxs[j]=np.argwhere(photonlist==photonlist[j])[0]
    plotexcitationdata[:,photonlist[j]]=excitationdata[0][:,int(idxs[j])]
    
# for j in range(len(excitationdata[0][0])):
#     plotexcitationdata[:,j] = plotexcitationdata[:,j]-np.mean(plotexcitationdata[:,j])
    
    
plt.figure()
plt.imshow(plotexcitationdata,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,wavelengthrange[1],wavelengthrange[0]],vmin=0)
plt.gca().invert_yaxis()
plt.colorbar()
plt.ylabel('Excitation Wavelength (nm)')
plt.xlabel('time (s)')
plt.title('Emission based excitation spectra ' + str(emwavelrange[idx][0]) + '- ' +str(emwavelrange[idx][1]) +'nm')  
#I basically plotted when the emission intensity is higher over certain wavelength regime to take that one over the other one. I do not see where these timeinterevals cross each other at the moment.


#each plotted in time frame
if Debugmode==True:
    fig,ax=plt.subplots(2,1,sharex=True,sharey=False) #plot excitation and emission above each other. If you want them to be visualized with the same window the yrange should cover the same distance. also zooming goes automatically
    im0=ax[0].imshow(Yfiltered,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths),np.min(wavelengths)],aspect='auto')
    # im0=ax[0].imshow(Yfiltered,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths),np.min(wavelengths)])
    ax[0].set_ylabel('Emission wavelength (nm)')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_title('Emission')
    ax[0].xaxis.set_tick_params(which='both', labelbottom=True)
    ax[0].invert_yaxis()
    fig.colorbar(im0,ax=ax[0])
    im1=ax[1].imshow(excitationdata[0],extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(excitationdata[1]),np.min(excitationdata[1][excitationdata[1]>0])],aspect='auto',vmin=0)
    # im1=ax[1].imshow(excitationdata[0],extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(excitationdata[1]),np.min(excitationdata[1][excitationdata[1]>0])])
    ax[1].set_ylabel('Excitation wavelength (nm)')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_title('Excitation')
    fig.colorbar(im1,ax=ax[1])
    ax[1].invert_yaxis()
    plt.tight_layout()
    
    #done to visualize same plot but just under Yfiltered indexes
    fig,ax=plt.subplots(2,1,sharex=True,sharey=False) #plot excitation and emission above each other. If you want them to be visualized with the same window the yrange should cover the same distance. also zooming goes automatically
    im0=ax[0].imshow(Yfiltered,extent=[0,len(Yfiltered[0]),np.max(wavelengths),np.min(wavelengths)],aspect='auto')
    # im0=ax[0].imshow(Yfiltered,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths),np.min(wavelengths)])
    ax[0].set_ylabel('Emission wavelength (nm)')
    ax[0].set_xlabel('Index')
    ax[0].set_title('Emission')
    ax[0].xaxis.set_tick_params(which='both', labelbottom=True)
    ax[0].invert_yaxis()
    fig.colorbar(im0,ax=ax[0])
    im1=ax[1].imshow(excitationdata[0],extent=[0,len(Yfiltered[0]),np.max(excitationdata[1]),np.min(excitationdata[1][excitationdata[1]>0])],aspect='auto',vmin=0)
    # im1=ax[1].imshow(excitationdata[0],extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(excitationdata[1]),np.min(excitationdata[1][excitationdata[1]>0])])
    ax[1].set_ylabel('Excitation wavelength (nm)')
    ax[1].set_xlabel('Index')
    ax[1].set_title('Excitation')
    fig.colorbar(im1,ax=ax[1])
    ax[1].invert_yaxis()
    plt.tight_layout()
#select excitation spectra based on trigger events from first photons in the emission cycle
#can do excitation emission correlation only on those. Forward and backward sweep combined

#%% excitation and emission correlation
# taus=[0,1,2,5,10,30,100]
# taus=[1,5,8,10,15]
taus=[1,2,5,10]
# taus = [5,10,100,400]
# taus=np.arange(1,40,1)
# plot = 'cov'
plot = 'corr'
# plot = 'none'
# Emissiondata1 = Yfiltered[:,0:100] #can get postselection by selecting only a particular window. Note that you need emissiondata to plot insteada of Yfiltered
# Emissiondata2 = binnedintensities[:,0:100]
emwavelrange=(605,645)
emwavelindices=(np.argmin(np.abs(wavelengths-emwavelrange[0])),np.argmin(np.abs(wavelengths-emwavelrange[1])))

emissionwavelengths = wavelengths[emwavelindices[0]:emwavelindices[1]]


excwavelrange=(500,592)
excwavelindices=(np.argmin(np.abs(excitationdata[1]-excwavelrange[0])),np.argmin(np.abs(excitationdata[1]-excwavelrange[1])))

excitationwavelengths = excitationdata[1][excwavelindices[0]:excwavelindices[1],1]

limlow = 600
limhigh = 800
Excitationdata = excitationdata[0][excwavelindices[0]:excwavelindices[1],limlow:limhigh]
Emissiondata = Yfiltered[emwavelindices[0]:emwavelindices[1],limlow:limhigh]


# emissioncovariance,emissionnormalization,emissioncorrelation = rplm.pearsoncorrelation(Excitationdata,Emissiondata,wavelengths,binnedwavelengths[:,0],taus=taus,plot=plot)
# select certain wavelengths pairs that were correlated in a line plot.
# most applicable to excitation and emission correlation

excemmcovariance,excemmnormalization,excemmcorrelation = rplm.pearsoncorrelation(Emissiondata,Excitationdata,excitationwavelengths,emissionwavelengths,taus=taus,plot=plot)
# excitationcovariance,excitationnormalization,excitationcorrelation = rplm.pearsoncorrelation(Excitationdata,Excitationdata,excitationwavelengths,excitationwavelengths,taus=taus,plot=plot)
emissioncovariance,emissionnormalization,emissioncorrelation = rplm.pearsoncorrelation(Emissiondata,Emissiondata,emissionwavelengths,emissionwavelengths,taus=taus,plot=plot)


#%% correlation of fits of spectra
fitted=rplm.fitspectra(Emissiondata,emissionwavelengths,0,1000,model='Gauss',Debugmode=False)

fitted[0][51]=fitted[0][50]
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

fitted[4][51]=fitted[4][50]
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

guessorigin=587
origin,rest = rplm.find_origin(emissionnormalization[1],guessorigin,testwavelengths)#,prominence=50,width=5)

# origin= 150
print(testwavelengths[origin])
datapoints = 80
origin= 129
# origin = np.where(emissioncovariance==np.amax(emissioncovariance))[2][0]  #because it is data which is sorted like : taus, wavelengths, wavelengths
print('polar transform centered around wavelength (nm)',wavelengths[origin])
datapoints = 50

correlationpolar = cartesiantopolarcorrelation(emissioncorrelation,origin,datapoints)
if Debugmode==True:
    plt.figure()
    plt.imshow(correlationpolar[1][1])
    plt.colorbar()
    plt.ylabel('radius (nm)')
    plt.title('Polar transform correlation map')

correlationpolarsum=np.sum(correlationpolar[1],axis=1)
correlationpolardata,covariancethetas = correlationpolar[1],correlationpolar[2] #use cartesian data as input

fourierdata = fouriertransform(correlationpolardata,covariancethetas)
fourierdata =fourierdata
fourierthetashift = fourierdata[2] #is shifted to make the plot look cleaner
fourierdatareal = fourierdata[3] # is shifted the same way as theta
fourierdataimag = fourierdata[4]

# plt.figure()
# plt.plot(fourierthetashift[1],fourierdataimag[1])
# plt.xlim(-6,6)

#%% here you can overlay guessed point with the actual plot to look for your maximum
from matplotlib.patches import Circle
circleradii = [1,2,3.5,5.5]
i=1
tau = taus[i]
fig,ax=plt.subplots()
# plt.figure()
ax.imshow(emissioncorrelation[i],extent=[np.min(wavelengths[emmwavmin:emmwavmax]),np.max(wavelengths[emmwavmin:emmwavmax]),np.max(wavelengths[emmwavmin:emmwavmax]),np.min(wavelengths[emmwavmin:emmwavmax])])
# plt.colorbar()
# plt.plot(wavelengths[181],1)
# r = Rectangle((30., 50.), 60., 50., edgecolor='yellow', facecolor='none')
# ax.scatter(wavelengths[origin],wavelengths[origin],edgecolor='red') #overlay plot with guessed position of the origin
# for j in range(len(circleradii)):
#     circle = Circle((wavelengths[origin],wavelengths[origin]),circleradii[j],fill=False)
#     # circle.set_facecolor('none')
#     ax.add_patch(circle)
# # plt.colorbar()
plt.gca().invert_yaxis()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Wavelength (nm)')
plt.title('Correlation map. tau = '+str(taus[i]))


#%% Plot fourier transforms
selectedradii = [5,10,20,30]
selectedradii= np.arange(0,50)
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

        for i in range(len(selectedradii)):
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
            plt.title('Fourier Transform imag. Tau = ' + str(tau) +' Wav. Org. = ' +str(testwavelengths[origin])[:5]+' nm')
            plt.xlim(-6,6)
            plt.legend(loc=1)

#here you can see how the amplitude of a particular n-fold cycle displays vs tau. Note that when you have a single tau defined you will get an empty plot (because there is only one point)

ftimag = rplm.Fouriercomponentvstau(fourierdataimag, fourierthetashift, selectedradii)
ftreal = rplm.Fouriercomponentvstau(fourierdatareal, fourierthetashift, selectedradii)

if Debugmode==True:
    plt.figure()
    plt.imshow(ftimag[1],extent=[np.min(selectedradii),np.max(selectedradii),np.max(taus),np.min(taus)])
    plt.xlabel('radius')
    plt.ylabel('tau')
    plt.title('Zeroth component imag')
    
    plt.figure()
    plt.imshow(ftimag[5],extent=[np.min(selectedradii),np.max(selectedradii),np.max(taus),np.min(taus)])
    plt.xlabel('radius')
    plt.ylabel('tau')
    plt.title('Second component imag')
    plt.colorbar()
    

if Debugmode==True:
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
plt.figure()
g2=rpl.MakeG2(times0,times1,dtmicro,g2restime=dtmicro*4,nrbins=1000)
plt.figure()
plt.plot(g2[0],g2[1])

histsettings=(440,600,3)
maxwavel=580
indices0=np.argwhere((calibcoeffs[1]+rplm.InVoltagenew(data[19]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0])<580)
exspec=rpl.Easyhist(calibcoeffs[1]+rplm.InVoltagenew(data[19][indices0]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
# plt.plot(exspec[0],(exspec[1]-np.min(exspec[1]))/np.sum(exspec[1]-np.min(exspec[1])))
indices1=np.argwhere((calibcoeffs[1]+rplm.InVoltagenew(data[20]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0])<580)
g2data=rpl.MakeG2(times0[indices0],times1[indices1],dtmicro,g2restime=dtmicro*4,nrbins=1000)
plt.figure()
plt.plot(g2data[0],g2data[1])
#%% Wavelength resolved autocorrelation
# this section gives the correlation with respect to time for a particular wavelength. One can take the for example the peak maximum
intensitycorr = intensitycorrelation(Yfiltered,Yfiltered,500)
delaytime = intensitycorr[1]
correlation = intensitycorr[5]
plt.figure()    
plt.plot(delaytime,correlation) 
plt.title('intensity correlation')
plt.xlabel('Delay')
plt.ylabel('Correlation (a.u.)')



js=[10,50,75,125,180,190,210,220] #these lines plot the correlation function for different times on selected wavelengths
for j in js:
    # print(wavelengths[j])
    wavelengthcorrelationdata = wavelengthcorrelation(Yfiltered[j],Yfiltered[j],500) #shows correlation of intensity levels at that particular wavelength. Is not normalized
    correlation = wavelengthcorrelationdata[0]
    delaytime = wavelengthcorrelationdata[1]
    plt.figure()
    plt.plot(delaytime,correlation)
    plt.xlabel('Delay')
    plt.ylabel('Correlation (a.u.)')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0)) #makes scientific notation for the axis. Pretty usefull when the normalization is way off and you want to compare the differences
    plt.title('Correlations value at wav ' + str(wavelengths[j]))

# plt.figure()
# plt.plot(np.arange(0,240,1/10),intensitycorr[2])
# plt.title('Sum emission intensity') #plots sum of intensities
# plt.ylabel('Intensity')
# plt.xlabel('Time (s)')

    
#%% Wiener process
#synthetic data starts here
centerstart,scaling,length = 600,1/40,500
wiener=rplm.wienerprocess(centerstart,scaling,length)
wiener2=(wiener-centerstart)*1.5+centerstart
wiener3=(wiener-centerstart)*2+centerstart
# wiener2=rplm.wienerprocess2(centerstart,scaling,length)

plt.figure()
# plt.plot(wiener);plt.plot(wiener2);plt.plot(wiener3)
plt.plot(wiener)
plt.xlabel('time')
plt.ylabel('Center wavelength (nm)')
plt.title('Wiener process')


#%%

def g(x,A,sigma,mu):
    return A / (sigma*np.sqrt(2*np.pi))*np.exp(-1 / 2 * (np.square((x-mu)/sigma)))

def randomsamp(mean,amplitude,sigma,wavelengths,nrphotons,bins):
    """ plot sampled gaussian curves with estimated wavelenghts"""
    x=wavelengths
    size=len(mean)
    sample=np.zeros((nrphotons,size))
    for i in tqdm(range(0,size)):    
        sample[:,i] = np.random.normal(mean[i],sigma[i],size=nrphotons)

    
    hist=np.zeros((bins,size))
    for i in range(0,size):        
        hist[:,i],x = np.histogram(sample[:,i],bins=bins,range=(np.min(x),np.max(x)))
        for j in range(bins):
            hist[j,i]= hist[j,i] +np.random.random(1)[0]/1e2

    x=x[:-1]+(x[1]-x[0])/2
    return x,sample,hist


def randomsamp2(mean,amplitude,sigma,wavelengths): 
    """ plot gaussian curves with estimated wavelenghts"""
    x=wavelengths
    size=len(mean)
    sample=np.zeros((len(x),size))
    
    samplenoise = np.zeros((len(x),size))
    for i in range(0,size):    
        sample[:,i] = g(x,amplitude[i],sigma[i],mean[i])
        for j in range(len(x)):
            
            samplenoise[j,i]=sample[j,i]+np.random.random(1)/100*3

    return x,sample,samplenoise

# mean = np.ones(600)*600
# for j in range(600):
#     if j%2==0:
#         mean[j]=600+np.random.random(1)[0]*2
#     else:
#         mean[j]=600.5+np.random.random(1)[0]*2
# # mean[300] = 601
# plt.figure()
# plt.plot(mean)
mean = np.copy(wiener)
# mean = np.copy(maxima)
# mean[mean>610]=608
# 
# mean[mean<605]=608
# mean = np.ones(len(mean))*600 # fixed means
# amplitude = np.abs(np.sum(Yfiltered,axis=0))/np.max(np.sum(Yfiltered,axis=0)) #uses the amplitude by summing the intensitieis for each time bin
amplitude = np.ones(len(mean)) #fixed amplitudes
# amplitude = (np.copy(wiener[1])-centerstart)*100
# sigma = np.zeros(len(mean))
sigma = np.ones(len(mean))*1/2
# sigma = np.random.random(len(mean))*8

testwavelengths=np.linspace(597,603,400)

test2 = randomsamp(mean,amplitude,sigma,testwavelengths,nrphotons=200,bins=600) 
# test2 = randomsamp(mean,amplitude,sigma,2000) 

testplots=[1,10,50,100]
plt.figure()
for i in range(len(testplots)):
    plt.plot(test2[0],test2[2][:,i])
plt.ylabel('Intensity')
plt.xlabel('Wavelength (nm)')
plt.title('Several plots')

#%% 1D correlation analysis on means
# pearson correlation on known and fitted spectra
originalmean = rplm.pearsoncorrelation1D(mean)
plt.figure()
plt.scatter(originalmean[3],originalmean[0])
plt.plot(originalmean[3],np.repeat(-originalmean[2],len(originalmean[0])),c='r');plt.plot(originalmean[3],np.repeat(originalmean[2],len(originalmean[0])),c='r');
plt.xlabel('Delay tau')
plt.ylabel('Correlation')
plt.title('Correlation of simulated maxima')


fitted=rpl.fitspectra(test2[2],test2[0],0,len(test2[2][0]),model='Gauss',Debugmode=False)
plt.figure()
plt.plot(fitted[2])
plt.xlabel('time')
plt.ylabel('Center wavelength (nm)')
plt.title('Fitted maxima')

# fitted=rpl.fitspectra(test2[2],test2[0],0,len(test2[2][0]),model='Gauss',Debugmode=False)
plt.figure()
plt.plot(fitted[2],label='Fitted maxima')
plt.plot(wiener,label='Wiener process')
plt.xlabel('time')
plt.ylabel('Center wavelength (nm)')
plt.title('Overlay')
plt.legend(loc=0)

fittedmean = rplm.pearsoncorrelation1D(fitted[2])
plt.figure()
plt.scatter(fittedmean[3],fittedmean[0])
plt.plot(fittedmean[3],np.repeat(-fittedmean[2],len(fittedmean[0])),c='r');plt.plot(fittedmean[3],np.repeat(fittedmean[2],len(fittedmean[0])),c='r');
plt.xlabel('Delay tau')
plt.ylabel('Correlation')
plt.title('Correlation of fitted maxima')
#%%

timeselectedmin=250
timeselectedmax=350
fittedmeanselected=rplm.pearsoncorrelation1D(mean[timeselectedmin:timeselectedmax])
plt.figure()
plt.plot(fittedmeanselected[3],fittedmeanselected[0])
plt.plot(fittedmeanselected[3],np.repeat(-fittedmeanselected[2],len(fittedmeanselected[0])),c='r');plt.plot(fittedmeanselected[3],np.repeat(fittedmeanselected[2],len(fittedmeanselected[0])),c='r');
plt.xlabel('Delay tau')
plt.ylabel('Correlation')
plt.title('Correlation of simulated maxima postselected ('+str(timeselectedmin)+' - ' +str(timeselectedmax)+')')


fittedmeanselected=rplm.pearsoncorrelation1D(fitted[2][timeselectedmin:timeselectedmax])
plt.figure()
plt.plot(fittedmeanselected[3],fittedmeanselected[0])
plt.plot(fittedmeanselected[3],np.repeat(-fittedmeanselected[2],len(fittedmeanselected[0])),c='r');plt.plot(fittedmeanselected[3],np.repeat(fittedmeanselected[2],len(fittedmeanselected[0])),c='r');
plt.xlabel('Delay tau')
plt.ylabel('Correlation')
plt.title('Correlation of fitted maxima postselected ('+str(timeselectedmin)+' - ' +str(timeselectedmax)+')')


#%% pearson correltaion map


taus =  [1,2,3,4]
# taus = np.arange(0,60,3)

# taus = [1,2,3,4,5]
# plot='all' #select this one to plot the actual plot you want to see
plot='corr'
# plot='none'
minwavspec1=np.argmin(np.abs(590-test2[0]))
maxwavspec1=np.argmin(np.abs(610-test2[0]))
# minwavspec1=593
# maxwavspec1=60
testwavelengths=test2[0][minwavspec1:maxwavspec1]
# plt.figure()  
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Intensity')
# plt.title('Simulated data')
# for i in range(size):
#     plt.plot(testwavelengths,test2[2][:,i])
timeend=len(mean) 
covnormcorr = rplm.pearsoncorrelation(test2[2][minwavspec1:maxwavspec1,:timeend],test2[2][minwavspec1:maxwavspec1,:timeend],testwavelengths,testwavelengths,taus=taus,plot=plot)
covnormcorr = rplm.pearsoncorrelation(test2[2][minwavspec1:maxwavspec1,timeselectedmin:timeselectedmax],test2[2][minwavspec1:maxwavspec1,timeselectedmin:timeselectedmax],testwavelengths,testwavelengths,taus=taus,plot=plot)
# plt.figure()
# plt.imshow(test2[2].T,extent=[np.min(testwavelengths),np.max(testwavelengths),len(mean),0],aspect='auto')
# plt.gca().invert_yaxis()
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('time (bins)')
# plt.title('Simulated data')

for i in range(len(taus)):
    tau=taus[i]
    # if tau==0:
    vmin=-0.2
    vmax0corr=0.3
    # print(vmax0corr)
    plt.figure()
    plt.imshow(covnormcorr[2][i],extent=[test2[0][minwavspec1],test2[0][maxwavspec1],test2[0][maxwavspec1],test2[0][minwavspec1]],vmin=vmin,vmax=vmax0corr)
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Wavelength (nm)')
    plt.title('Correlation map. tau = '+str(taus[i]))

    # if savefig=True: #attempt to save figures on the fly
    # plt.savefig('E:/Martijn/ETH/results/20200310_PM111_specdiffusion/QD2/Correlation_map_tau'+str(tau)+'_excitation',dpi=800) 
            
# plt.figure()  
# for i in range(size):
#     plt.plot(testwavelengths,test[2][:,i])    
# covnormcorr = rplm.pearsoncorrelation(test[2],test[2],testwavelengths,testwavelengths,taus=taus,plot=plot)
# covnormcorr = spectralcorrelation(test2[1],test2[1],testwavelengths,testwavelengths,taus=taus,plot=plot)

covariancedata = covnormcorr[0]
normalizationdata = covnormcorr[1]
correlationdata = covnormcorr[2]
# covnormcorr = spectralcorrelation(test2[1],test2[1],testwavelengths,testwavelengths,taus=taus,plot=plot)

if Debugmode==True: # some impression of the data plotted without the plot= stuff
    plt.figure()
    plt.imshow(covariancedata[0],extent=[np.min(testwavelengths),np.max(testwavelengths),np.max(testwavelengths),np.min(testwavelengths)])
    plt.gca().invert_yaxis()
    plt.colorbar()
    
    
    plt.figure()
    plt.imshow(normalizationdata[0],extent=[np.min(testwavelengths),np.max(testwavelengths),np.max(testwavelengths),np.min(testwavelengths)])
    plt.gca().invert_yaxis()
    plt.colorbar()
    
        
    plt.figure()
    plt.imshow(correlationdata[5],extent=[np.min(testwavelengths),np.max(testwavelengths),np.max(testwavelengths),np.min(testwavelengths)])
    plt.gca().invert_yaxis()
    plt.colorbar()


    #%% origin computation
#based on normalization map and the deviation thereinnn

# normalizationdata = normalizationdata[0]
guessorigin=600
origin,rest = rplm.find_originsyntheticdata(normalizationdata[0],guessorigin,testwavelengths,prominence=10,width=1)

# origin= 150
print(testwavelengths[origin])
datapoints = 80
#%% fourier transform on synthetic data
correlationpolar = cartesiantopolarcorrelation(correlationdata,origin,datapoints)
correlationpolardata,correlationthetas= correlationpolar[1],correlationpolar[2]
# correlationpolarsum=np.sum(correlationpolar[1],axis=1)
fourierdatacorrelation = fouriertransform(correlationpolardata,correlationthetas)
# fourierdatacorrelation = fouriertransform(correlationpolarsum,correlationthetas)
fourierdata =fourierdatacorrelation
fourierthetashift = fourierdata[2] #is shifted to make the plot look cleaner
fourierdatareal = fourierdata[3] # is shifted the same way as theta
fourierdataimag = fourierdata[4]

if Debugmode==True:
    spectrum=1
    plt.figure()
    plt.imshow(correlationpolar[1][spectrum],extent=[np.min(correlationthetas)*180/np.pi,np.max(correlationthetas)*180/np.pi,len(correlationpolardata[0]),0])
    plt.colorbar()
    plt.ylabel('radius')
    plt.xlabel('theta')
    plt.title('Polar transform correlation map. Tau = ' +str(taus[spectrum]))

    plt.figure()
    plt.imshow(correlationpolarsum)
    plt.colorbar()
#%%selectedradii = [5,20,40,60] # select a particular radius
selectedradii = [5,10,20,30]
# selectedradii= np.arange(0,50)
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
    for j in range(len(fourierdata[0])):
    
    # for j in range(0,1):
        tau = taus[j]
        plt.figure()
        spectrum = j # this parameter is for different taus

        for i in range(len(selectedradii)):
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
            plt.title('Fourier Transform imag. Tau = ' + str(tau) +' Wav. Org. = ' +str(testwavelengths[origin])[:5]+' nm')
            plt.xlim(-6,6)
            plt.legend(loc=1)

#here you can see how the amplitude of a particular n-fold cycle displays vs tau. Note that when you have a single tau defined you will get an empty plot (because there is only one point)

ftimag = rplm.Fouriercomponentvstau(fourierdataimag, fourierthetashift, selectedradii)
ftreal = rplm.Fouriercomponentvstau(fourierdatareal, fourierthetashift, selectedradii)

if Debugmode==True:
    plt.figure()
    plt.imshow(ftimag[1],extent=[np.min(selectedradii),np.max(selectedradii),np.max(taus),np.min(taus)])
    plt.xlabel('radius')
    plt.ylabel('tau')
    plt.title('Zeroth component imag')
    
    plt.figure()
    plt.imshow(ftimag[5],extent=[np.min(selectedradii),np.max(selectedradii),np.max(taus),np.min(taus)])
    plt.xlabel('radius')
    plt.ylabel('tau')
    plt.title('Second component imag')
    plt.colorbar()
    

if Debugmode==True:
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

#%% Fitted data emission spectra
# Fits emission spectra based on timeaveraged spectrum and then their individuals

fits=rplm.fitspectra(Yfiltered[210:662],wavelengths[210:662],0,500,Debugmode=False)

    #%% correlation analysis on fits
beginsel=0
endsel=100
maximasel = maxima[beginsel:endsel]
pearcorr= rplm.pearsoncorrelation1D(maximasel)
plt.figure()
plt.plot(maximasel)#-np.mean(maximasel))

plt.figure()
plt.plot(pearcorr[0])
plt.plot(range(len(pearcorr[0])),np.ones(len(pearcorr[0]))*pearcorr[2],c='r')
plt.plot(range(len(pearcorr[0])),-np.ones(len(pearcorr[0]))*pearcorr[2],c='r')


#%% cuts in correlation map
#done to show how similar photon with wavelength lambda1 is to photons with wavelength lambda2 is for different taus.
test = emissioncorrelation
wavinterest = [119,120,121,125,250]

for j in range(len(wavinterest)):
    wavselect = wavinterest[j]
    plt.figure()
    plt.imshow(test[:,wavselect,:],extent=[np.min(wavelengths),np.max(wavelengths),np.max(taus),np.min(taus)])
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.xlabel('Emission wavelength (nm)')
    plt.ylabel('Delay time')
    plt.title('Correlation map. wav =' +str(wavelengths[wavselect])+'nm.')
#%% cuts in correlation map
#start making cuts in the excitation peaks and the emission peaks of the correlation map/.
tauidx=0
tau = taus[tauidx]
# tau = 8
plt.figure()
plt.plot(wavelengths,emissioncorrelation[tauidx,:,excitationindex])
plt.title('Em Exc correlation (Ex.wav = ' + str(binnedwavelengths[:,1][excitationindex])[:7]+' nm). Tau = ' + str(tau))
plt.ylabel('correlation')
plt.xlabel('Emission wavelength (nm)')
plt.ylim(-0.1,0.25)


plt.figure()
plt.imshow(emissioncorrelation[:,:,excitationindex],extent=[np.min(wavelengths),np.max(wavelengths),np.max(taus),np.min(taus)])
plt.title('Em Exc correlation (Ex.wav = ' + str(binnedwavelengths[:,1][excitationindex])[:7]+' nm)')
plt.ylabel('Delay time tau')
plt.xlabel('Emission wavelength (nm)')
plt.gca().invert_yaxis()
plt.colorbar()
# plt.ylim(-0.1,0.25)

emissionindex = 65
emissionindex = 71
emissionindex = 80
wavelengths[emissionindex]

plt.figure()
plt.plot(binnedwavelengths[:,1],emissioncorrelation[tauidx,emissionindex,:])
plt.title('Em Exc correlation (Emm.wav = ' + str(wavelengths[emissionindex])[:7]+' nm) Tau = ' + str(tau))
plt.ylabel('correlation')
plt.xlabel('Excitation wavelength (nm)')
plt.ylim(-0.1,0.25)





          
#%%
reffolder2 = 'E:/Martijn/ETH/results/20200310_PM111_specdiffusion/'
calibspec=rpl.importASC(reffolder2+'backref_4p87MHz_200Hz_500mVpp_off60mV_redone.asc') #this is the file of the reference spectrum.
refspec=np.mean(calibspec[0][:,490:510],1)*1.2398/calibspec[1]

refspec=refspec-np.mean(refspec[925:len(refspec)])
refspec=refspec/np.max(refspec)
refspec=savgol_filter(refspec, 23, 3)
interprefspec= interp1d(calibspec[1], refspec,kind='cubic',fill_value='extrapolate')
if Debugmode==True:
    plt.figure()
    plt.plot(calibspec[1],refspec,Wavelengthspec,interprefspec(Wavelengthspec),'.')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.legend(['Measured backreflection','Evaluated at measurement points'])
#% Plot raw data against reference spec
    fig,ax1 = plt.subplots()
    ax1.plot(Wavelengthspec,ylistspec/max(ylistspec),'b')
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Intensity', color='b')
    ax1.tick_params('y', colors='b')
    #ax1.set_ylim([0,0.11])
    ax2 = ax1.twinx()
    ax2.plot(calibspec[1],refspec,'r')
    ax2.set_ylabel('Excitation Intensity (nm)', color='r')
    ax2.tick_params('y', colors='r')
    #plt.plot(calibspec[1],refspec)#,calibspec[1],interprefspec(calibspec[1]))
    plt.xlim([400,630])
    plt.xlabel('Excitation Wavelength (nm)')
    plt.ylabel('Reflected Intensity')
    plt.legend(['Measured spectrum','Backref of excitation'])


#% Plot corrected spectrum
plt.figure()
plt.plot(Wavelengthspec,ylistspec/interprefspec(Wavelengthspec))


plt.xlim([np.min(Wavelengthspec),np.max(Wavelengthspec)])
plt.ylim([0,np.max(ylistspec/interprefspec(Wavelengthspec))])
#plt.xlim([510,595])
#plt.ylim((0, 6000))

plt.xlabel('Excitation Wavelength (nm)')
plt.ylabel('Intensity')
plt.title(namelist[0])

#% Plot dispersion PLE measurement
#RefPLE=importPLE(folder+'PM151_dil_Em622_Ex300-600_step1_Dwell0.3_ExBW5_EmBW0.3_Refcor_Emcor.txt')
#fig,ax1 = plt.subplots()
#ax1.plot(Wavelengthspec,ylistspec/interprefspec(Wavelengthspec),'b')
#ax1.set_xlabel('Excitation Wavelength (nm)')
#ax1.set_ylabel('Film Micro PLE', color='b')
#ax1.set_xlim([430,650])
#ax1.set_ylim([0,0.3*np.max(ylistspec)])
#ax1.ticklabel_format(style='sci',scilimits=(-2,3),axis='both')
#ax1.tick_params('y', colors='b')
#ax2 = ax1.twinx()
#ax2.plot(RefPLE[0],RefPLE[1],'r')
#ax2.set_ylim([0,1.5e6])
#ax2.set_ylabel('Dispersion PLE', color='r')
#ax2.tick_params('y', colors='r')
#ax2.ticklabel_format(style='sci',scilimits=(-2,3),axis='both')
#ax1.set_title('"Dot" 2')           
                

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


#%% Import other measurements
# Linewidthspec=rpl.importASC(folder+'Refspec_n50mV.asc')
# Linewidthspec[0]=np.mean(Linewidthspec[0][:,509:577],1)
# Linewidthspec[0]=Linewidthspec[0]-471
# Linewidthspec[0]=Linewidthspec[0]/np.max(Linewidthspec[0])
# plt.plot(1240/Linewidthspec[1],Linewidthspec[0])
# plt.xlim((2.195,2.205))

#%%
# PLspec=rpl.importASC(folder+'QD2_4p8MHr_200Hz_500mVpp_60mVoff_pluscamera.ptu')
# PLspec[0]=np.mean(PLspec[0][:,509:577],1)
# PLspec[0]=PLspec[0]-471*5
# #plt.figure()
# plt.plot(1240/PLspec[1],PLspec[0]/np.max(PLspec[0]))
# # plt.xlim((2.18,2.215))
# plt.ylim((0,1.1))
# # refspec=np.mean(calibspec[0][:,457:466],1)*1.2398/calibspec[1]
# # plt.figure()
# # plt.plot(PLspec[0])
#%% Excitation resolved decay
plotrange=[500,590]
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
plt.gca().invert_yaxis()
plt.colorbar()

#actually the only difference with normalization or not is that the intensity at lower times are more/less pronounced



#%% WIth wavelength binning
# fig, ax = plt.subplots(constrained_layout=True)
# wavelList=calibcoeffs[1]+InVoltage(data[19]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
# [ylistspec,xlistspec] = np.histogram(wavelList,600) # 600 is the ratio between the laser repetition rate and the Galvomirror changing in one time interval. Example: 200Hz and 1.95 MHz results in 608 times the laser scans in one galvomirror cycle. This number is a bit wrong -> change frequency galvomirror
# tlistspec = (xlistspec[:-1]+0.5*(xlistspec[1]-xlistspec[0]))
# rng=range(110,520)

# # ax.plot(1240/Wavelengthspec,ylistspec/interprefspec(Wavelengthspec))
# ax.plot(1240/tlistspec[rng],ylistspec[rng]/interprefspec(tlistspec[rng])/np.max(ylistspec[rng]/interprefspec(tlistspec[rng])))
# ax.plot(1240/PLspec[1],PLspec[0]/np.max(PLspec[0]))
# ax.plot((1240/512,1240/512),(0,0.3),'--k')
# # plt.xlim((2.31,2.75))
# ax.set_ylim((0,1.2))
# ax.set_xlabel('Energy (eV)')
# ax.set_ylabel('Intensity')
# ax.set_title(namelist[0])
# secax = ax.secondary_xaxis('top', functions=(nmtoeV, eVtonm))
# secax.set_xlabel('Wavelength (nm)')
# # plt.plot(1240/Linewidthspec[1],5000*Linewidthspec[0])
# ax.legend(('PLE','PL'))

#%% Plot spec 2 against 3
# fig, ax = plt.subplots(constrained_layout=True)


# # ax.plot(1240/Wavelengthspec,ylistspec/interprefspec(Wavelengthspec))
# ax.plot(tlistspec2[rng2],ylistspec2[rng2]/interprefspec(tlistspec2[rng2])/np.max(ylistspec2[rng2]/interprefspec(tlistspec2[rng2])))
# ax.plot(PLspec2[1],PLspec2[0]/np.max(PLspec2[0]))
# ax.plot(tlistspec3[rng3],ylistspec3[rng3]/interprefspec(tlistspec3[rng3])/np.max(ylistspec3[rng3]/interprefspec(tlistspec3[rng3])))
# ax.plot(PLspec3[1],PLspec3[0]/np.max(PLspec3[0]))
# plt.xlim((520,640))
# ax.set_ylim((0,1.2))
# ax.set_xlabel('Wavelength (nm)')
# ax.set_ylabel('Intensity')
# secax = ax.secondary_xaxis('top', functions=(eVtonm,nmtoeV))
# secax.set_xlabel('Energy (eV)')
# # plt.plot(1240/Linewidthspec[1],5000*Linewidthspec[0])
# ax.legend(('PLE Dot2','PL Dot2','PLE Dot3','PL Dot3'))
#%% Find peaks cww
rng=range(110,520)
testdata=ylistspec[rng]/interprefspec(tlistspec[rng])
testwavel=tlistspec[rng]
result=scipy.signal.find_peaks_cwt(testdata,np.arange(5,20))
plt.figure()
plt.plot(1240/testwavel,testdata)
for k in range(len(result)):
    plt.plot((1240/testwavel[result[k]],1240/testwavel[result[k]]),(np.min(testdata),testdata[result[k]]),'--k')
    plt.text(1240/testwavel[result[k]],testdata[result[k]]+10000,str(round(1240/testwavel[result[k]],3)),horizontalalignment='center',verticalalignment='center')
plt.xlabel('Excitation Energy (eV)')
# plt.ylim((0,1300))
plt.xlim((2.35,2.83))


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
plt.hist2d(data[3]*data[1],calibcoeffs[1]+InVoltagenew_c(data[19]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],range=[[1,Texp],[530,590]],bins=[120,160])#,norm=mcolors.LogNorm())
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


exwavel = calibcoeffs[1]+InVoltage(data[19]*data[1],Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
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


#%% Spectral diffusion
#binwidthspecdiff=0.5
#limitsblue = rpl.HistPhotons(timesblue*dtmicro,binwidthspecdiff,Texp)
#Meanexwavel=np.zeros(len(limitsblue)-1)
#nphotons = np.zeros(len(limitsblue)-1)
#for tbinnr in tqdm(range(len(limitsblue)-1)):
#    macrocycle=macrotimescycleblue[limitsblue[tbinnr]:limitsblue[tbinnr+1]]
#    nphotons[tbinnr]=limitsblue[tbinnr+1]-limitsblue[tbinnr]
#    Exwavels=calibcoeffs[1]+InVoltage(macrocycle*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
#    Meanexwavel[tbinnr]=np.mean(Exwavels)
#plt.figure()
##plt.hist2d(data[3]*data[1],calibcoeffs[1]+InVoltage(data[19]*data[1],Freq,Vpp,Voffset,Verror)*calibcoeffs[0],range=[[1,Texp],[520,590]],bins=[2400,40])#,norm=mcolors.LogNorm())
##plt.xlabel('Time (s)')
##plt.ylabel('Excitation wavelength (nm)')
##plt.colorbar()
#Meanexwavel=np.nan_to_num(Meanexwavel)
#wmean=np.sum(Meanexwavel*nphotons)/np.sum(nphotons)
#plt.subplot(211)
#plt.title(namelist[0])
#plt.plot(np.arange(len(limitsblue)-1)*binwidthspecdiff,nphotons)
#plt.ylabel('Photons in feature')
#plt.xlabel('Time (s)')
#plt.subplot(212)
#plt.plot(np.arange(len(limitsblue)-1)*binwidthspecdiff,Meanexwavel)
#plt.ylim([wmean-1,wmean+1])
#plt.ylabel('Peak position')
#plt.plot([0,Texp],[wmean,wmean],'--k')
#
#%% SPectral diffusion 2
plt.figure()
plt.hist2d(data[3]*data[1],calibcoeffs[1]+InVoltage(data[19]*data[1],Freq,Vpp,Voffset,Verror)*calibcoeffs[0],range=[[1,Texp],[520,590]],bins=[2400,40])#,norm=mcolors.LogNorm())
plt.xlabel('Time (s)')
plt.ylabel('Excitation wavelength (nm)')
plt.colorbar()
plt.plot(np.arange(len(limitsblue)-1)*binwidthspecdiff,Meanexwavel,'w')
#%% Plot excitation gated decay
# plt.figure()
MaxLikelihoodFunction_c = nb.jit(nopython=True)(MaxLikelihoodFunction)
fitdata=GetLifetime(microtimesblue,dtmicro,dtmacro,250e-9,tstart=-1,plotbool=True,ybg=-1,method='ML_c')
# plt.xlim([18,100])
# plt.set_yscale('log')

#%% Lifetime vs Intensity
MaxLikelihoodFunction_c = nb.jit(nopython=True)(MaxLikelihoodFunction)
#macrotimesin=data[3]
#microtimesin=data[2]
microtimesin=microtimesblue
macrotimesin=macrotimesblue
macrotimescyclein=macrotimescycleblue
binwidth=0.02
macrolimits = rpl.HistPhotons(macrotimesin*dtmacro,binwidth,Texp)
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
lifetime=GetLifetime(microtimesin,dtmicro,dtmacro,100e-9,tstart=-1,histbinmultiplier=1,ybg=-1,plotbool=True,method='ML_c')
plt.show()
#[taulist,Alist,ybglist] = Parallel(n_jobs=-1, max_nbytes=None)(delayed(processInput)(tbinnr) for tbinnr in tqdm(range(nrtbins-1)))
#plt.figure()
#test=np.zeros((wavelbins,len(limits)-1))
for tbinnr in tqdm(range(len(limits)-1)):
    microtimes = microtimesin[limits[tbinnr]:limits[tbinnr+1]]
    macrotimescycle = macrotimescyclein[limits[tbinnr]:limits[tbinnr+1]]
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
    #     [taulist[tbinnr],Alist[tbinnr],ybglist[tbinnr],buff[tbinnr]]=GetLifetime(microtimes,dtmicro,dtmacro,60e-9,tstart=lifetime[3]*dtmicro,histbinmultiplier=1,ybg=0*lifetime[2]*binwidth/Texp,plotbool=False,method='ML_c') 
    # else:
    #     [taulist[tbinnr],Alist[tbinnr],ybglist[tbinnr],buff[tbinnr]]=GetLifetime(microtimes,dtmicro,dtmacro,20e-9,tstart=lifetime[3]*dtmicro,histbinmultiplier=1,ybg=0*lifetime[2]*binwidth/Texp,plotbool=False,method='ML_c') 

    tauav[tbinnr]=(np.mean(microtimes)-lifetime[3])*dtmicro*1e9
    photonspbin[tbinnr]=len(microtimes)
    #using th
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
ax2.set_ylim([0,40])
ax2.set_ylabel('Lifetime (ns)', color='r')
ax2.tick_params('y', colors='r')
#ax2.ticklabel_format(style='sci',scilimits=(-2,3),axis='both')
ax1.set_title(namelist[0])


#%% Lifetime correlation
# this idea is based on not using the GetLifeTime function to generate lifetimes, but instead make a 2D correlation map with the observed microtime (and macrotime), separated by a time interval of dT
shortset = 1000 #not analyzing all the data to save time during checks
microtimes = data[2][0:shortset]*dtmicro*1e9 #nanoseconds
macrotimes = data[3][0:shortset]*dtmacro #seconds

dt = 0.05 #seconds
# dt = microtimes[1]-microtimes[0] #seconds
# timerange = 1e-1
dt = macrotimes[1]-macrotimes[0] #seconds
timerange = 1e-4


# krange = int(shortset/100)
krange = shortset
macrotimesindex = np.zeros((len(macrotimes),krange))
# macrotimesindex = np.zeros(len(macrotimes))

for j in tqdm(range(len(macrotimes))):
    for k in range(krange):
        if dt-timerange<=macrotimes[j]-macrotimes[k]<=dt+timerange:
        # if dt-timerange<=microtimes[j]-microtimes[k]<=dt+timerange:
            macrotimesindex[j,k]+=1 #shows up hits when at position j,k the condition was satisfied
            #makes something where it kind of searches where the j parameter is a t the moment
plt.figure()
plt.imshow(macrotimesindex,aspect='auto')
plt.gca().invert_yaxis()
            
microtimes1 = np.zeros(len(macrotimes))
microtimes2 = np.zeros(len(macrotimes))

for j in tqdm(range(len(macrotimes))): #might wanna combine this step with the step above so that you dont have to generate such a large array (because that is not going to be able to be handled by your pc). also think of something where you can decrease the size of the array by selectively looking for photons in the range where the j loop is already
    for k in range(krange):
        if macrotimesindex[j,k]==1:
            microtimes1[j] = microtimes[j] #search for microtimes when condition is met
            microtimes2[j] = microtimes[k]
            
microtimes1sor = sorted(microtimes1) #sort them
microtimes2sor = sorted(microtimes2)

hist,xedge,yedge = np.histogram2d(microtimes1sor,microtimes2sor,bins=20)
            


plt.figure()
plt.imshow(hist,extent=[np.min(xedge),np.max(xedge),np.max(yedge),np.min(yedge)]) #shows correlation map of correlated part, witht axis meaning delay time (not lifetime). Lifetime is generated based on laplace transform of this stuff
plt.gca().invert_yaxis()
plt.colorbar()

#now thingss get ugly in the sense that you now have to do this laplace stuff

#%% Lifetime correlation
# this idea is based on not using the GetLifeTime function to generate lifetimes, but instead make a 2D correlation map with the observed microtime (and macrotime), separated by a time interval of dT
shortset = 10000 #not analyzing all the data to save time during checks
microtimes = (data[2][0:shortset]-lifetime[3])*dtmicro*1e9 #nanoseconds
macrotimes = data[3][0:shortset]*dtmacro #seconds

# dt = 0.05 #seconds
# dt = microtimes[1]-microtimes[0] #seconds
# timerange = 1e-1
# dt = macrotimes[1]-macrotimes[0] #seconds
# timerange = 1e-4
tmin=0
tmax=1/10

krange = int(shortset/100)

microtimes1 = np.zeros(len(macrotimes))
microtimes2 = np.zeros(len(macrotimes))
microtimes2 = np.zeros((len(macrotimes),krange,krange))


# for j in tqdm(range(6)): #idea was sort of to not let k run through all the values but only a selected part to increase the speed. It works.
#     # if j < krange:
#     for k in range(10):
#         if tmin<=macrotimes[j]-macrotimes[k]<=tmax:
#             print("found a hit")

for j in tqdm(range(len(macrotimes))): #idea was sort of to not let k run through all the values but only a selected part to increase the speed. It works.
    if j < krange:
        for k in range(krange):
            if tmin<=macrotimes[j]-macrotimes[k]<=tmax:
                # print("found a hit")
            # if dt-timerange<=macrotimes[j]-macrotimes[k]<=dt+timerange:
            # if dt-timerange<=microtimes[j]-microtimes[k]<=dt+timerange:
                microtimes1[j] = microtimes[j] #search for microtimes when condition is met
                microtimes2[j][k][k] = microtimes[k]
    elif j+krange<len(macrotimes):
        for k in range(j-krange,j+krange):
            if tmin<=macrotimes[j]-macrotimes[k]<=tmax:
            # if dt-timerange<=macrotimes[j]-macrotimes[k]<=dt+timerange:
            # if dt-timerange<=microtimes[j]-microtimes[k]<=dt+timerange:
                microtimes1[j] = microtimes[j] #search for microtimes when condition is met
                microtimes2[j] = microtimes[k]
                
    else:
        for k in range(j-krange,len(macrotimes)):
            if tmin<=macrotimes[j]-macrotimes[k]<=tmax:
            # if dt-timerange<=macrotimes[j]-macrotimes[k]<=dt+timerange:
    # if dt-timerange<=microtimes[j]-microtimes[k]<=dt+timerange:
                microtimes1[j] = microtimes[j] #search for microtimes when condition is met
                microtimes2[j] = microtimes[k]
            
                
microtimes1sor = sorted(microtimes1) #sort them
microtimes2sor = sorted(microtimes2)

hist,xedge,yedge = np.histogram2d(microtimes1sor,microtimes2sor,bins=200)
            


plt.figure()
plt.imshow(hist,extent=[np.min(xedge),np.max(xedge),np.max(yedge),np.min(yedge)]) #shows correlation map of correlated part, witht axis meaning delay time (not lifetime). Lifetime is generated based on laplace transform of this stuff
plt.gca().invert_yaxis()
plt.colorbar()

#now thingss get ugly in the sense that you now have to do this laplace stuff
#%%
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

            
#%%Plot selected spectra
#plt.figure()
#reftoplot=np.sum(Exspeclist,1)
##for j in range(floor(Texp/10)-1):
##toplot=np.sum(Exspeclist[:,j*10:(j+1)*10],1)
#toplot=np.sum(Exspeclist[:,0:30],1)
#plt.plot(tlistex,toplot/np.sum(toplot))#-reftoplot/np.sum(reftoplot))
#toplot=np.sum(Exspeclist[:,125:165],1)
#plt.plot(tlistex,toplot/np.sum(toplot))#-reftoplot/np.sum(reftoplot))
#toplot=np.sum(Exspeclist[:,218:230],1)
#plt.plot(tlistex,toplot/np.sum(toplot))#-reftoplot/np.sum(reftoplot))
#plt.xlabel('Wavelength (nm)')
#plt.ylabel('Excitation spectrum')
#plt.legend(["0 to 30 s","125 to 165s","176 to 200s"])

#%% Plot histogram
plt.figure()
histrange=(0,np.max(photonspbin))
plt.hist(photonspbin,int(np.max(photonspbin))+1,histrange)
plt.xlabel('Photons per '+str(int(binwidth*1000))+' ms bin')
plt.ylabel('Occurence')
#simcounts=np.random.poisson(lam=285,size=8000)
#plt.hist(simcounts,int(np.max(simcounts))+1,(0,np.max(simcounts)))
#simcounts=np.random.poisson(lam=40,size=100)
#plt.hist(simcounts,int(np.max(simcounts))+1,(0,np.max(simcounts)))
#plt.ylim([1000,1600])
#%% Plot FLID map
plt.figure()
plt.hist2d(tauav,photonspbin,(50,int(np.max(photonspbin)/2)),range=[[0,50],[0,np.max(photonspbin)]],norm=mcolors.LogNorm())
plt.title('FLID map')
plt.ylabel('Counts per bin')
plt.xlabel('Lifetime (ns)')
plt.colorbar()

#plt.plot(np.arange(0,50),np.arange(0,50)*505/27+8,'w')

#%% Intensity limits
limitex=180
limittrlow=40
limittrhigh=75
microtimesin=microtimesblue
macrotimesin=macrotimesblue
macrotimescyclein=macrotimescycleblue
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
for tbinnr in range(len(limits)-1):
    nphots = limits[tbinnr+1]-limits[tbinnr]
    microtimes = microtimesin[limits[tbinnr]:limits[tbinnr+1]]
    macrotimescycle = macrotimescycleblue[limits[tbinnr]:limits[tbinnr+1]]
    if nphots>limitex:
        microtimes_ex[nrex:nrex+nphots]=microtimes
        macrotimescycle_ex[nrex:nrex+nphots]=macrotimescycle
        bins_ex+=1
        nrex+=nphots

    elif nphots>limittrlow and nphots<limittrhigh and taulist[tbinnr]<15:# and taulist[tbinnr]>0.05: #and (photonspbin[tbinnr]-7)/taulist[tbinnr]>112/28
#    elif photonspbin[tbinnr]>limittrlow and photonspbin[tbinnr]<limittrhigh and taulist[tbinnr]>0.5 and (photonspbin[tbinnr]-10)/taulist[tbinnr]>505/27:
        microtimes_trion[nrtrion:nrtrion+nphots]=microtimes
        macrotimescycle_trion[nrtrion:nrtrion+nphots]=macrotimescycle
        bins_trion+=1
        nrtrion+=nphots
    elif photonspbin[tbinnr]>limittrlow and photonspbin[tbinnr]<limitex:# and tauav[tbinnr]>6 :
        microtimes_mid[nrmid:nrmid+nphots]=microtimes
        macrotimescycle_mid[nrmid:nrmid+nphots]=macrotimescycle
        bins_mid+=1
        nrmid+=nphots
    elif nphots<limittrlow and taulist[tbinnr]<3:
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
#% Lifetime of Exciton and Trion
plt.figure()
fitex=GetLifetime(microtimes_ex,dtmicro,dtmacro,100e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_ex/(Texp/binwidth),method='ML_c')
fittrion=GetLifetime(microtimes_mid,dtmicro,dtmacro,10e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_mid/(Texp/binwidth),method='ML_c')
fittrion=GetLifetime(microtimes_trion,dtmicro,dtmacro,10e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_trion/(Texp/binwidth),method='ML_c')

# fitoff=GetLifetime(microtimes_off,dtmicro,dtmacro,10e-9,tstart=-1,plotbool=True,ybg=0*lifetime[2]*bins_trion/(Texp/binwidth),method='ML_c')
#fitmid=GetLifetime(microtimes_mid,dtmicro,dtmacro,10e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_mid/(Texp/binwidth),method='ML_c')

print('Rad lifetime ratio:'+str(fittrion[1]/bins_trion/(fitex[1]/bins_ex)))
#plt.xlim([0,220])
plt.legend(['High cps','High cps fit','Mid cps','Mid cps fit','Low cps','Low cps fit'])

#%% Proper wavelength binning
macrotimescycle_exnew=macrotimescycle_ex[0:len(macrotimescycle_trion)-1]
Wavelengthex=calibcoeffs[1]+InVoltage(macrotimescycle_ex*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
Wavelengthtrion=calibcoeffs[1]+InVoltage(macrotimescycle_trion*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0] #for 250Hz
Wavelengthoff=calibcoeffs[1]+InVoltage(macrotimescycle_off*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
Wavelengthmid=calibcoeffs[1]+InVoltage(macrotimescycle_mid*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
histrange=(np.min(Wavelengthex),np.max(Wavelengthex))
wavelbins=200
[ylistex,xlistex] = np.histogram(Wavelengthex,wavelbins,histrange)
tlistex = (xlistex[:-1]+0.5*(xlistex[1]-xlistex[0]))
[ylisttrion,xlisttrion] = np.histogram(Wavelengthtrion,wavelbins,histrange)
tlisttrion = (xlisttrion[:-1]+0.5*(xlisttrion[1]-xlisttrion[0]))
[ylistoff,xlistoff] = np.histogram(Wavelengthoff,wavelbins,histrange)
tlistoff = (xlistoff[:-1]+0.5*(xlistoff[1]-xlistoff[0]))
[ylistmid,xlistmid] = np.histogram(Wavelengthmid,wavelbins)
tlistmid = (xlistmid[:-1]+0.5*(xlistmid[1]-xlistmid[0]))
plt.figure()
plt.plot(tlistex,ylistex*(np.sum(ylisttrion)/np.sum(ylistex))/interprefspec(tlistex))
plt.plot(tlistmid,ylistmid*(np.sum(ylisttrion)/np.sum(ylistmid))/interprefspec(tlistmid)-5000)
plt.plot(tlisttrion,ylisttrion/interprefspec(tlisttrion)-10000)
# plt.plot(tlistoff,ylistoff*(np.sum(ylisttrion)/np.sum(ylistoff)))
#dotsex,=plt.plot(tlistex,ylistex*(np.sum(ylisttrion)/np.sum(ylistex)),'.')
# dotsmid,=plt.plot(tlistmid,ylistmid*(np.sum(ylistmid)/np.sum(ylistmid)),'.')
#plt.plot(tlistex,scipy.ndimage.filters.gaussian_filter1d(ylistex*(np.sum(ylisttrion)/np.sum(ylistex)),2),color=dotsex.get_color())

#dotsoff,=plt.plot(tlistoff,ylistoff*(np.sum(ylisttrion)/np.sum(ylistoff)),'.')
#plt.plot(tlistoff,ylistoff*(np.sum(ylisttrion)/np.sum(ylistoff)))
#dotsex,=plt.plot(tlistmid,ylistmid/interprefspec(tlistmid),'.')
#plt.plot(tlistmid,scipy.ndimage.filters.gaussian_filter1d(ylistmid/interprefspec(tlistmid),5))#,color=dotsex.get_color())

#dotstrion,=plt.plot(tlisttrion,ylisttrion,'.')#/max(ylistspec))
#plt.plot(tlisttrion,scipy.ndimage.filters.gaussian_filter1d(ylisttrion*1.,2),color=dotstrion.get_color())

#plt.plot(tlistex,ylisttrion-ylistex*(np.sum(ylisttrion)/np.sum(ylistex)),'.')
#plt.plot(tlistoff,ylistoff*(np.sum(ylisttrion)/np.sum(ylistoff))-ylistex*(np.sum(ylisttrion)/np.sum(ylistex)),'.')
#plt.plot(tlistex,scipy.ndimage.filters.gaussian_filter1d(ylisttrion,4)-scipy.ndimage.filters.gaussian_filter1d(ylistex*(np.sum(ylisttrion)/np.sum(ylistex)),4))
#plt.plot(tlistex,ylistmid*(np.sum(ylisttrion)/np.sum(ylistmid))-ylistex*(np.sum(ylisttrion)/np.sum(ylistex)))
#plt.plot(tlistex,np.cumsum(ylistex)*(np.sum(ylisttrion)/np.sum(ylistex)))#/interprefspec(tlistex))#/max(ylistspec))
#plt.plot(tlisttrion,np.cumsum(ylisttrion))#/max(ylistspec))
#plt.plot(tlistex,np.cumsum(ylisttrion)-np.cumsum(ylistex)*(np.sum(ylisttrion)/np.sum(ylistex)))#/interprefspec(tlistex))#/max(ylistspec))
plt.xlim([445,wavellimit])
plt.ylim((-10000,50000))
#plt.ylim([0,20])
plt.xlabel('Excitation Wavelength (nm)')
plt.ylabel('(normalized) Counts')
plt.legend(['High','Medium','Low'])

#%% Improper wavel binning
histrange=(np.min(data[19])*dtmacro,np.max(data[19])*dtmacro)
wavelbins=150
[ylistex,xlistex] = np.histogram(macrotimescycle_ex*dtmacro,wavelbins,histrange)
tlistex = (xlistex[:-1]+0.5*(xlistex[1]-xlistex[0]))
[ylisttrion,xlisttrion] = np.histogram(macrotimescycle_trion*dtmacro,wavelbins,histrange)
tlisttrion = (xlisttrion[:-1]+0.5*(xlisttrion[1]-xlisttrion[0]))
[ylistoff,xlistoff] = np.histogram(macrotimescycle_off*dtmacro,wavelbins,histrange)
tlistoff = (xlistoff[:-1]+0.5*(xlistoff[1]-xlistoff[0]))


Wavelengthex=calibcoeffs[1]+InVoltage(tlistex,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
Wavelengthtrion=calibcoeffs[1]+InVoltage(tlisttrion,Freq,Vpp,Voffset,Verror)*calibcoeffs[0] #for 250Hz
Wavelengthoff=calibcoeffs[1]+InVoltage(tlistoff,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]


plt.figure()
plt.plot(Wavelengthex,ylistex*(np.sum(ylisttrion)/np.sum(ylistex)))
#plt.plot(tlistmid,ylistmid*(np.sum(ylisttrion)/np.sum(ylistmid)))
plt.plot(Wavelengthtrion,ylisttrion,'.')
plt.plot(Wavelengthoff,ylistoff*(np.sum(ylisttrion)/np.sum(ylistoff)),'.')
#dotsex,=plt.plot(tlistex,ylistex*(np.sum(ylisttrion)/np.sum(ylistex)),'.')
#dotsmid,=plt.plot(tlistmid,ylistmid*(np.sum(ylistmid)/np.sum(ylistmid)),'.')
#plt.plot(tlistex,scipy.ndimage.filters.gaussian_filter1d(ylistex*(np.sum(ylisttrion)/np.sum(ylistex)),2),color=dotsex.get_color())

#dotsoff,=plt.plot(tlistoff,ylistoff*(np.sum(ylisttrion)/np.sum(ylistoff)),'.')
#plt.plot(tlistoff,ylistoff*(np.sum(ylisttrion)/np.sum(ylistoff)))
#dotsex,=plt.plot(tlistmid,ylistmid/interprefspec(tlistmid),'.')
#plt.plot(tlistmid,scipy.ndimage.filters.gaussian_filter1d(ylistmid/interprefspec(tlistmid),5))#,color=dotsex.get_color())

#dotstrion,=plt.plot(tlisttrion,ylisttrion,'.')#/max(ylistspec))
#plt.plot(tlisttrion,scipy.ndimage.filters.gaussian_filter1d(ylisttrion*1.,2),color=dotstrion.get_color())

#plt.plot(tlistex,ylisttrion-ylistex*(np.sum(ylisttrion)/np.sum(ylistex)),'.')
#plt.plot(tlistoff,ylistoff*(np.sum(ylisttrion)/np.sum(ylistoff))-ylistex*(np.sum(ylisttrion)/np.sum(ylistex)),'.')
#plt.plot(tlistex,scipy.ndimage.filters.gaussian_filter1d(ylisttrion,4)-scipy.ndimage.filters.gaussian_filter1d(ylistex*(np.sum(ylisttrion)/np.sum(ylistex)),4))
#plt.plot(tlistex,ylistmid*(np.sum(ylisttrion)/np.sum(ylistmid))-ylistex*(np.sum(ylisttrion)/np.sum(ylistex)))
#plt.plot(tlistex,np.cumsum(ylistex)*(np.sum(ylisttrion)/np.sum(ylistex)))#/interprefspec(tlistex))#/max(ylistspec))
#plt.plot(tlisttrion,np.cumsum(ylisttrion))#/max(ylistspec))
#plt.plot(tlistex,np.cumsum(ylisttrion)-np.cumsum(ylistex)*(np.sum(ylisttrion)/np.sum(ylistex)))#/interprefspec(tlistex))#/max(ylistspec))
plt.xlim([wavellimitlow,wavellimit])
#plt.ylim([0,20])
plt.xlabel('Excitation Wavelength (nm)')
plt.ylabel('(normalized) Counts')
plt.legend(['Ex','Trion','Off'])
#%% Plot normalized spectra
plt.figure()
plt.plot(tlistex,ylistex*(np.sum(ylisttrion)/np.sum(ylistex))/interprefspec(tlistex))
plt.plot(tlisttrion,ylisttrion/interprefspec(tlisttrion))
#plt.ylim([0,1400])
plt.legend(['Bright','Gray'])
plt.xlabel('Excitation Wavelength (nm)')
plt.ylabel('normalized Counts')
plt.title(namelist[0])
#%% Plot difference spectrum
plt.figure()
plt.plot(tlisttrion,scipy.ndimage.filters.gaussian_filter1d(ylisttrion,5)-ylistex*(np.sum(ylisttrion)/np.sum(ylistex)))
#plt.plot(tlistmid,scipy.ndimage.filters.gaussian_filter1d(ylistmid*(np.sum(ylisttrion)/np.sum(ylistmid)),5)-ylistex*(np.sum(ylisttrion)/np.sum(ylistex)))

#%%Lifetime sorting

microtimesfast = np.zeros(len(data[2]),dtype='int64')
macrotimesfast = np.zeros(len(data[2]),dtype='int64')
macrotimescyclefast = np.zeros(len(data[2]),dtype='int64')
nrfast = 0
microtimesslow = np.zeros(len(data[2]),dtype='int64')
macrotimesslow = np.zeros(len(data[2]),dtype='int64')
macrotimescycleslow = np.zeros(len(data[2]),dtype='int64')
timesslow = np.zeros(len(data[2]),dtype='int64')
nrslow = 0
wavellimit = 590
startmicro = lifetime[3]
delaylimit = lifetime[3]+round(3/dtmicro*1e-9)
delayend = round(40/dtmicro*1e-9)
delayend = lifetime[3]+round(10/dtmicro*1e-9)

for j in tqdm(range(len(data[2]))):
    if data[2][j] > startmicro and data[2][j] < delaylimit:
        microtimesfast[nrfast] = data[2][j]
        macrotimesfast[nrfast] = data[3][j]
        macrotimescyclefast[nrfast] = data[19][j]
        nrfast += 1
    elif data[2][j] > delaylimit and data[2][j] < delayend:
        microtimesslow[nrslow] = data[2][j]
        macrotimesslow[nrslow] = data[3][j]
        macrotimescycleslow[nrslow] = data[19][j]
        timesslow[nrslow]=times0[j]
        nrslow += 1
microtimesfast = microtimesfast[:nrfast]
macrotimesfast = macrotimesfast[:nrfast]
macrotimescyclefast = macrotimescyclefast[:nrfast]
microtimesslow = microtimesslow[:nrslow]
macrotimesslow = macrotimesslow[:nrslow]
macrotimescycleslow = macrotimescycleslow[:nrslow]
wavelbins=300
[ylistspecfast,xlistspecfast] = np.histogram(macrotimescyclefast,wavelbins)
tlistspecfast = (xlistspecfast[:-1]+0.5*(xlistspecfast[1]-xlistspecfast[0]))*dtmacro
Wavelengthspecfast=calibcoeffs[1]+InVoltage(tlistspecfast,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
[ylistspecslow,xlistspecslow] = np.histogram(macrotimescycleslow,wavelbins)
tlistspecslow = (xlistspecslow[:-1]+0.5*(xlistspecslow[1]-xlistspecslow[0]))*dtmacro
Wavelengthspecslow=calibcoeffs[1]+InVoltage(tlistspecslow,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
plt.figure()
plt.plot(Wavelengthspecfast,ylistspecfast)#/np.sum(ylistspecfast))
plt.plot(Wavelengthspecslow,ylistspecslow)#/np.sum(ylistspecslow))#+0.2*np.max(ylistspecslow)/np.sum(ylistspecslow))
plt.legend(['Fast','Slow'])
#%%

plt.plot(Wavelengthspecfast,ylistspecfast/1.55)#/np.sum(ylistspecfast))
plt.plot(Wavelengthspecslow,ylistspecslow)#/np.sum(ylistspecslow))#+0.2*np.max(ylistspecslow)/np.sum(ylistspecslow))
plt.legend(['Fast','Slow'])
#%% Intensity slices
aggregatedphot=0
plt.figure()
ylistbinspec=np.zeros([300,20])
Wavelengthbinspec=np.zeros([300,20])
[ylistspecav,xlistspecav] = np.histogram(data[19],300)
for binnr in range(nrbins): # for each intensity bin
    # select photons in bins with intensity within Ilims(binnr) and Ilims(binnr+1)
    if binnr>-1:
        histbinmultiplier=16
        [binmicrotimes0,bintimes0,binmacrotimescycle0,binmicrotimes1,bintimes1,binmacrotimescycle1,onbincount] = SliceHistogram(microtimes0,times0,limits0,data[19],microtimes1,times1,limits1,data[20],dtmicro,dtmacro,Ilims[binnr],Ilims[binnr+1])  
#        [ylist,xlist] = np.histogram(binmicrotimes0,int(dtmacro/(dtmicro*histbinmultiplier)),[0,int(dtmacro/dtmicro)])    
#        tlist = (xlist[:-1]+0.5*(xlist[1]-xlist[0]))*dtmicro*1e9 # convert x-axis to time in ns
        [ylistbinspec[:,binnr],xlistbinspec] = np.histogram(binmacrotimescycle0,300)
        tlistbinspec = (xlistbinspec[:-1]+0.5*(xlistbinspec[1]-xlistbinspec[0]))*dtmacro
        [ylistbinspec[:,binnr],xlistbinspec] = np.histogram(binmacrotimescycle0,300)
        tlistbinspec = (xlistbinspec[:-1]+0.5*(xlistbinspec[1]-xlistbinspec[0]))*dtmacro
        Wavelengthbinspec[:,binnr]=calibcoeffs[1]+InVoltage(tlistbinspec,Freq,Vpp,Voffset,Verror)*calibcoeffs[0] #for 250Hz
        plt.plot(Wavelengthbinspec[:,binnr],ylistbinspec[:,binnr]/np.sum(ylistbinspec[:,binnr],0)+1*0.0015*binnr)
        # plt.plot(Wavelengthbinspec[:,binnr],ylistbinspec[:,binnr]/np.sum(ylistbinspec[:,binnr])-ylistspecav/np.sum(ylistspecav)+0.005*binnr)
        plt.xlim([520,592])
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Photons per bin')
        photinbin=np.sum(ylistbinspec[:,binnr])
        aggregatedphot+=photinbin
#        istart=0
#        iend=max(tlist)
#        nrnewbins=int(dtmacro/(dtmicro*histbinmultiplier))
#        bgcounts=binwidth*onbincount*80
#        bgcpb=bgcounts/nrnewbins
#        initParams = [np.max(ylist), 25]
##        [Abinlist[measnr,binnr],Alphabinlist[measnr,binnr],Thetabinlist[measnr,binnr]]=MaxLikelihoodFit_c(tlist,ylist,istart,iend,bgcpb,initParams,plotbool=True)
#        GetLifetime(binmicrotimes0,dtmicro,dtmacro,20e-9,tstart=lifetime[3]*dtmicro,histbinmultiplier=1,ybg=1+0*lifetime[2]*binwidth/Texp,plotbool=True,method='ML_c') 
#    # calculate average lifetimes
##        plt.figure(15)
#        plt.title(['binnr:', str(binnr),'Alpha: ', str(round(Alphabinlist[measnr,binnr],2)),'Theta: ', str(round(Thetabinlist[measnr,binnr]*180/np.pi))])
    
    
#        [tauavelist[measnr,binnr],Alist[measnr,binnr],_] = GetLifetime(np.append(binmicrotimes0,binmicrotimes1),dtmicro,dtmacro,dttau,-1,histbinmultiplier,ybg*onbincount)
        print('onbincount:',onbincount)
        print('Overall photons:', photinbin)






#%% Other plots and legacy
#%%
totalPhots=np.zeros(100)
correlation=np.zeros(100)
correlationtrion=np.zeros(100)
timepassed=np.zeros(100)
timepassed_trion=np.zeros(100)
for j in range(0,100):
    macrotimescycle_exnew=macrotimescycle_ex[0:(100*(j+1)-1)]
    macrotimescycle_trionnew=macrotimescycle_trion[0:(10*(j+1)-1)]
    timepassed[j]=macrotimescycle_exnew[-1]-macrotimescycle_exnew[0]
    timepassed_trion[j]=macrotimescycle_trionnew[-1]-macrotimescycle_trionnew[0]
    totalPhots[j]=50*(j+1)
    Wavelengthmid=calibcoeffs[1]+InVoltage(macrotimescycle_exnew*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
    Wavelengthtrion=calibcoeffs[1]+InVoltage(macrotimescycle_trionnew*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0]
    [ylistmid,xlistmid] = np.histogram(Wavelengthmid,wavelbins)
    [ylisttrion,xlisttrion] = np.histogram(Wavelengthtrion,wavelbins)
    a=np.corrcoef(ylistex,ylistmid)
    b=np.corrcoef(ylistex,ylisttrion)
    correlation[j]=a[0,1]
    correlationtrion[j]=b[0,1]
plt.plot(totalPhots,np.cumsum(timepassed)*dtmacro)
#plt.plot(timepassed_trion,correlationtrion)
#plt.figure()
#plt.plot(totalPhots,timepassed)
#plt.plot(totalPhots,timepassed_trion)
#%%G2 eval
test=MakeG2(times0,times1,dtmicro,g2restime=dtmicro,nrbins=8000)
#%%
g2axis=test[0]
g2vals=test[1]
centerindex=np.argmin(np.abs(g2axis))
avwindow=round(dtmacro/dtmicro/4)
npeaks=floor(abs(g2axis[0])*2*1e-9/dtmacro)
peakarea=np.zeros(npeaks,dtype=float)
peakcenter=np.zeros(npeaks,dtype=float)
for j in range(npeaks):
    peakcenter[j]=np.round(centerindex+(j-(npeaks-1)/2)*dtmacro/dtmicro)
    peakarea[j]=np.sum(g2vals[int(np.round(peakcenter[j]-avwindow/2)):int(np.round(peakcenter[j]+avwindow/2))])
#    peakarea[j]=np.sum(g2vals)
plt.plot(g2axis[peakcenter.astype(int)],peakarea/np.mean(peakarea),'*')
plt.ylim([0,1.2*np.max(peakarea/np.mean(peakarea))])


#%% Filtering
#plt.plot(ylistex)
#Subtratc baseline
baseline=(ylisttrion[-1]-ylisttrion[0])/(tlisttrion[-1]-tlisttrion[0])*(tlisttrion-tlisttrion[0])+ylisttrion[0]
testspec=ylisttrion-baseline
#plt.plot(tlisttrion,ylisttrion,tlisttrion,baseline)
freq=np.fft.rfftfreq(tlisttrion.shape[-1])
#testfourier=np.fft.fft(testspec)
fourierreal=np.fft.rfft(testspec)
testnobasel=np.fft.rfft(ylisttrion)
Intfourierreal=np.square(np.absolute(fourierreal))
Fourieramp=np.absolute(fourierreal)
variance=np.var(fourierreal[round(len(fourierreal)/2):-1])
Noise=2*variance/(wavelbins/2)
plt.semilogy(Intfourierreal/Noise)
#plt.legend(['baseline subtracted','Raw'])
#plt.xlim([])

#%% PCA 
#idea was to represent data with variances, where variance 1 is due to intensity fluctautions and variance 2 due to SD
#I just took this example from the internet. At the moment having doubts whether to go for this or not. because in taht paper they did this k-means clustering based on individual orhotognal components, but also their modelling on orthogoanl compoentns. not sure whether this gives a false positive biased result
from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
np.random.seed(42)

X_digits, y_digits = load_digits(return_X_y=True)
data = scale(X_digits)

n_samples, n_features = data.shape
n_digits = len(np.unique(y_digits))
labels = y_digits

sample_size = 300

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


print(82 * '_')
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')


def bench_k_means(estimator, name, data):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=n_digits, n_init=10),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=n_digits).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
              name="PCA-based",
              data=data)
print(82 * '_')

# #############################################################################
# Visualize the results on PCA-reduced data

reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

# Step size of the mesh. Decrease to increase the quality of the VQ.
h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

# Plot the decision boundary. For that, we will assign a color to each
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtain labels for each point in mesh. Use last trained model.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot the centroids as a white X
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

#%% Close all open figures in all scripts
plt.close('all')

#%% savefig


plt.savefig('E:/Martijn/ETH/results/20200310_PM111_specdiffusion/QD2/Correlation_map_tau0',dpi=800)
