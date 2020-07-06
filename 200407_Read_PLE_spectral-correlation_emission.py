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



# In[25]:


### GENERAL EVALUATION OF TTTR data     

# parameters (use forward slashes!)
Debugmode=False     #Set to true to make plots in the individual sections of the code to check that everything is working properly

basefolders={'DESKTOP-BK4HAII':'C:/Users/rober/Documents/Doktorat/Projects/SingleParticle_PLE/Andor Spectrometer/',
             'HP_Probook':'E:/Martijn/ETH/results/',
             'mavt-omel-w004w':'E:/LAB_DATA/Robert/'} #dictionary which base folder to use on which computer. For you this probably would be everything until 
folder = basefolders[socket.gethostname()]+'20200630_CdSe_cryo_HR2/'

filename = 'HR2_QD6_9p734MHz_200Hz_260mVpp_n20mVoff_ND0'

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
plt.figure()
data = rpl.ImportT3(folder + filename + '.out',HHsettings)
# data = rpl.ImportT3(folder + filename + '.ptu',HHsettings)
# data = rpl.ImportT3(folder + filename + '.ptu',HHsettings)
Texp = round(data[8]*data[1]*1024,0) # macrotime values can store up to 2^10 = 1024 before overflow. overflows*1024*dtmacro gives experiment time [s]
print('averaged cps on det0 and det1:',np.array(data[6:8])/Texp)
print('experimental time in s:',Texp)
[microtimes0,microtimes1,times0,times1,dtmicro,dtmacro,decaytlist,decayylist] = rpl.ShiftPulsedData(data[2],data[4],data[3],data[5],data[0],data[1]) #decaytlist and decayylist are the two variables you want to check for the modulation trace
# MaxLikelihoodFunction_c = nb.jit(nopython=True)(rpl.MaxLikelihoodFunction)
lifetime=rpl.GetLifetime(microtimes0,dtmicro,dtmacro,dtfit=400e-9,tstart=-1,histbinmultiplier=1,ybg=-1,plotbool=True,expterms=2,method='ML_c')

limits0 = rpl.HistPhotons(times0*dtmicro,binwidth,Texp) #gives index of first photon of det0 in each bin
limits1 = rpl.HistPhotons(times1*dtmicro,binwidth,Texp)

# make an intensity trace and find Imax
inttrace = rpl.MakeIntTrace(limits0,limits1,binwidth,Texp)


#%% Wavelength calibration
Voltage=np.array([-80,-40,0,40,80])
Wavelength_calib=np.array([578.6,560.2,541.5,523,504.9])
calibcoeffs=np.polyfit(Voltage,Wavelength_calib,1)

if Debugmode==True:
    plt.figure()
    plt.scatter(Voltage,Wavelength_calib)
    plt.xlabel('Voltage')
    plt.ylabel('Excitation wavelength (nm)')
    plt.plot(Voltage,calibcoeffs[1]+Voltage*calibcoeffs[0],Voltage,Wavelength_calib,'*')
    plt.legend(['Linear Fit','Measurement'])
    
Vpp=float(filename.split('mVpp_')[0].split('_')[-1])
Freq=float(filename.split('Hz_')[-2])
Voffset=0
if filename.split('mVoff')[0].split('Vpp_')[-1].startswith('n'):
    Voffset=float(filename.split('mVoff')[0].split('Vpp_')[-1].split('n')[-1])
else:
    Voffset=float(filename.split('mVoff')[0].split('Vpp_')[-1])
# Voffset=0
matchrange=(500,570)
tshift=rplm.Findtshift(Freq,Vpp,Voffset,calibcoeffs,data[19],dtmacro,matchrange,(-6e-4,-2e-4)) #coarse sweep
tshift=rplm.Findtshift(Freq,Vpp,Voffset,calibcoeffs,data[19],dtmacro,matchrange,(tshift-1e-5,tshift+1e-5),Debugmode=True)

#%% Excitation wavelength camera data
#seems to match quite well. Can also do later a correction for this, but it would only matter like 0.5-1 nm or so.
file1='backref_9p734MHz_0Hz_0mVpp_20mVoff_ND0.asc'
backref1=rpl.importASC(folder+file1)  
# plt.figure()
# plt.imshow(backref1[0],extent=[0,len(backref1[0]),backref1[1][0],backref1[1][-1]])

max1=print(backref1[1][np.unravel_index(backref1[0].argmax(),backref1[0].shape)[1]])
print(calibcoeffs[1]+calibcoeffs[0]*float(file1.split('mVoff')[0].split('_')[-1]))

file1='backref_9p734MHz_0Hz_0mVpp_60mVoff_ND0.asc'
backref1=rpl.importASC(folder+file1)  
# plt.figure()
# plt.imshow(backref1[0],extent=[0,len(backref1[0]),backref1[1][0],backref1[1][-1]])

max1=print(backref1[1][np.unravel_index(backref1[0].argmax(),backref1[0].shape)[1]])
print(calibcoeffs[1]+calibcoeffs[0]*float(file1.split('mVoff')[0].split('_')[-1]))
#%% laser intensity corrected
    #done by sweeping laser spectrum over camera and divide QD em spectrum over normalized reference spectrum
reffile = str(filename.split('MHz')[0].split('_')[-1])
reffolder = folder
calibspec = rpl.importASC(reffolder+'backref_9p734MHz_200Hz_500mVpp_60mVoff_ND0.asc')
# calibspec = rpl.importASC(reffolder+'backref'+str(reffile)+'MHz_200Hz_500mVpp_60mVoffset.asc')

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
plt.figure()
# # plt.plot(np.linspace(500,600,300),interprefspec(np.linspace(500,600,300)))
plt.plot(calibspec[1],refspec)
plt.title('Spectrum of laser')
plt.xlabel('Excitation wavelength (nm)')
plt.ylabel('Intensity')



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
wavelengthrangemin,wavelengthrangemax = 400,590
wavelengthrange = (wavelengthrangemin,wavelengthrangemax)
# wavelengthrange = matchrange
histogrambin=247
[excspec,excspecwavtemp] = np.histogram(InVoltagenew_c(dtmacro*data[19],Freq,Vpp,Voffset,tshift)*calibcoeffs[0]+calibcoeffs[1],histogrambin,range=wavelengthrange)
excspecwav = [excspecwavtemp[:-1]+(excspecwavtemp[1]-excspecwavtemp[0])/2]
excspeccorr = excspec/interprefspec(excspecwav)

plt.figure()
plt.plot(excspecwav[0],excspec)    
plt.title('Uncorrected') 
plt.xlabel('Excitation wavelength (nm)')
plt.ylabel('Intensity')
plt.figure()
plt.plot(excspecwav[0],excspeccorr[0])   
plt.title('Laser corrected') 
plt.xlabel('Excitation wavelength (nm)')
plt.ylabel('Intensity')


    
    
#%% Generate binned excitation spectra
#this generates part of unbinned excitation spectra. It does not resolve for forward and backward scan
binning = int(Texp/2) #binning is in units of seconds (s)
# binning = 1/10
expplottime = data[17][-1]*dtmacro
excitationbegin = data[17][0]*dtmacro
# binning = 1 #binning is in units of seconds (s)
wavelengthrangemin,wavelengthrangemax = 400,593
wavelengthrange = (wavelengthrangemin,wavelengthrangemax)
# wavelengthrange  = matchrange
histogrambin=50 #number is irrelevant for excitation and emission correlation. Fixed that.
# histogrambin= int(np.divide((wavelengthrangemax-wavelengthrangemin),(np.max(Wavelengthspec)-np.min(Wavelengthspec)))*1/dtmacro/200/40) #200 is frequency of glvo cycle
# firstphotonlist = rpl.HistPhotons(times0*dtmicro,binning,Texp)
firstphotonlist = rpl.HistPhotons(times0*dtmicro,binning,expplottime)
binnedintensities = np.zeros((histogrambin,len(firstphotonlist)))
binnedwavelengths = np.zeros((histogrambin,len(firstphotonlist)))



    
for i in range(len(firstphotonlist)-1):
    [intensitiestemp,wavelengthstemp] = np.histogram(InVoltagenew_c(dtmacro*data[19][firstphotonlist[i]:firstphotonlist[i+1]],Freq,Vpp,Voffset,tshift)*calibcoeffs[0]+calibcoeffs[1],histogrambin,range=wavelengthrange)
    wavelengthstempavg = (wavelengthstemp[:-1]+0.5*(wavelengthstemp[1]-wavelengthstemp[0]))
    binnedwavelengths[:,i]= wavelengthstempavg
    binnedintensities[:,i]=intensitiestemp/interprefspec(wavelengthstempavg)
    
# vmax=np.max(binnedintensities)
# plt.figure()
# plt.imshow(binnedintensities,extent=[excitationbegin,expplottime,np.max(binnedwavelengths[:,1]),np.min(binnedwavelengths[:,1])],vmax=vmax)
# plt.gca().invert_yaxis()
# plt.ylabel('Excitation wavelength (nm)')
# plt.xlabel('time (s)')
# plt.title('binning = ' +str(binning))
# plt.colorbar()    

if Debugmode==True:
    plt.figure()
    plt.plot(binnedwavelengths,binnedintensities)
    plt.xlim(np.min(np.delete(binnedwavelengths.ravel(),np.where(binnedwavelengths.ravel()<=1))),np.max(binnedwavelengths))
    plt.xlabel('Excitation Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title('binning = ' +str(binning) +'s')
    # plt.title('Excitation spectrum')
    
    
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
        plt.plot(wavelengths,timeaverage.ravel())
        plt.xlabel('Emission wavelength (nm)')
        plt.ylabel('Intensity')
        plt.title('time-averaged signal')
        
        emwavelrange=(480,530)
        emwavelindices=(np.argmin(np.abs(wavelengths-emwavelrange[0])),np.argmin(np.abs(wavelengths-emwavelrange[1])))
        timeminplot=0
        timemaxplot=-1
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


taus = np.arange(0,20,1)
# taus=[0,5,10,100]
# taus=[0,1,2]

# taus=[1]
# taus=[1,2,10,15]
# plot = 'corr'
# plot='norm'
# plot='all'
# plot = 'cov'
plot='none'

# wavmin = 80
# wavelengths[wavmin]

# wavmax = 300
# wavelengths[wavmax]

timemin = 700
timemax = 900
Emissiondata1 = Yfiltered[:,timemin:timemax] #can get postselection by selecting only a particular window. Note that you need emissiondata to plot insteada of Yfiltered
Emissiondata2 = Yfiltered[:,timemin:timemax]

# Emissiondata1 = Yfiltered[wavmin:wavmax,15:23] #can get postselection by selecting only a particular window. Note that you need emissiondata to plot insteada of Yfiltered
# Emissiondata2 = Yfiltered[wavmin:wavmax,15:23]

# Emissiondata1 = Yfiltered[:,15:23] #can get postselection by selecting only a particular window. Note that you need emissiondata to plot insteada of Yfiltered
# Emissiondata2 = Yfiltered[:,15:23]
# emissioncovariance,emissionnormalization,emissioncorrelation = rplm.pearsoncorrelation(Yfiltered,Yfiltered,wavelengths,wavelengths,taus=taus,plot=plot)
# emissioncovariance,emissionnormalization,emissioncorrelation = spectralcorrelation(Yfiltered,Yfiltered,wavelengths,wavelengths,taus=taus,plot=plot)

emissioncovariance,emissionnormalization,emissioncorrelation = rplm.pearsoncorrelation(Emissiondata1,Emissiondata2,wavelengths,wavelengths,taus=taus,plot=plot)



for i in range(10,11):
    tau=taus[i]
    # if tau==0:
    # vmin=-0.2
    # vmax0corr=0.3
    # print(vmax0corr)
    plt.figure()
    plt.imshow(emissioncorrelation[i],extent=[np.min(wavelengths),np.max(wavelengths),np.max(wavelengths),np.min(wavelengths)])
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Wavelength (nm)')
    plt.title('Correlation map. tau = '+str(taus[i]))

test = emissioncorrelation
# wavinterest = [119,120,121,125,250]
emwavelrange=[607,607.5,608,608.5,610]
for j in range(len(emwavelrange)):
    # wavselect = wavinterest[j]
    wavselect=np.argmin(np.abs(wavelengths-emwavelrange[j]))
    vmax=0.4
    plt.figure()
    plt.imshow(test[:,wavselect,:],extent=[np.min(wavelengths),np.max(wavelengths),np.max(taus),np.min(taus)],vmax=vmax,aspect='auto')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.xlabel('Emission wavelength (nm)')
    plt.ylabel('Delay time')
    plt.title('Correlation map. wav =' +str(wavelengths[wavselect])+'nm.')
    
guessorigin=607
origin,rest = rplm.find_origin(emissionnormalization[1],guessorigin,wavelengths)#,prominence=50,width=5)

print('index in list: ',origin, ' wavelength in plot: ',wavelengths[origin])


test = emissioncorrelation
# wavinterest = [119,120,121,125,250]
emwavelrange=[607.61]
for j in range(len(emwavelrange)):
    # wavselect = wavinterest[j]
    wavselect=np.argmin(np.abs(wavelengths-emwavelrange[j]))
    vmax=0.4
    plt.figure()
    plt.imshow(test[:,wavselect,:],extent=[np.min(wavelengths),np.max(wavelengths),np.max(taus),np.min(taus)],vmax=vmax,aspect='auto')
    plt.gca().invert_yaxis()
    plt.colorbar()
    plt.xlabel('Emission wavelength (nm)')
    plt.ylabel('Delay time')
    plt.title('Correlation map. wav =' +str(wavelengths[wavselect])+'nm.')
    

emwavelrange=(606,609.5)
emwavelindices=(np.argmin(np.abs(wavelengths-emwavelrange[0])),np.argmin(np.abs(wavelengths-emwavelrange[1])))
croppedwavelengths=wavelengths[emwavelindices[0]:emwavelindices[1]]
searchedtaus=taus
croppedtest=test[:,emwavelindices[0]:emwavelindices[1],emwavelindices[0]:emwavelindices[1]]
maxima=np.zeros((len(searchedtaus),2),dtype='int64')
for j in range(len(searchedtaus)):
    # maxima[j] = np.argmax(test[j,emwavelindices[0]:emwavelindices[1],emwavelindices[0]:emwavelindices[1]])
    maxima[j,:] = np.unravel_index(croppedtest[j].argmax(),croppedtest[j].shape)
wavmax0=croppedwavelengths[maxima[:,0]]
wavmax1=croppedwavelengths[maxima[:,1]]

plt.figure()
plt.plot(wavmax0);plt.plot(wavmax1)

fitted=rplm.fitspectra(Emissiondata1,wavelengths,0,len(Emissiondata1[0]),model='Lor',Debugmode=False)
fitted=rplm.fitspectra(Emissiondata1,wavelengths,0,1,model='Lor',Debugmode=True)

origin=np.zeros(len(taus))
guessorigin=607
for j in range(len(taus)):
    
    origin[j],rest= rplm.find_origin(emissionnormalization[j],guessorigin,wavelengths)#,prominence=50,width=5)

# print('index in list: ',origin, ' wavelength in plot: ',wavelengths[origin])
#%% Generate binned excitation spectra for excitation emission correlation
# #select particular exctiation wavelength based on chosen range emission wavelengths


rawdata = data[25]
originaldata= data[19]
selecteddata=data[25]
wavelengthrange = matchrange
wavelengthrange = (450,590)
histogram = 300
excitationdata = rplm.histogramexcitationspectra(Freq,Vpp,Voffset,tshift,calibcoeffs,dtmacro,rawdata,selecteddata,originaldata,wavelengthrange,histogram)
# excitationdata = histogramexcitationspectranorm(rawdata,originaldata,wavelengthrange,histogram,interprefspec)

# for j in range(len(excitationdata[0][0])):
#     excitationdata[0][:,j] = excitationdata[0][:,j]-np.mean(excitationdata[0][:,j])

# maxexcitation = np.max(excitationdata[0])    
vmax=np.max(excitationdata[0])
plt.figure()
plt.imshow(excitationdata[0],extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,wavelengthrange[1],wavelengthrange[0]],vmin=0,vmax=vmax,aspect='auto')
plt.gca().invert_yaxis()
plt.colorbar()
plt.ylabel('Excitation Wavelength (nm)')
plt.xlabel('time (s)')
plt.title('Full excitation spectra')  

#%% fit only part of emission and then overplot with sampled function
emwavelrange=(605,640)
emwavelindices=(np.argmin(np.abs(wavelengths-emwavelrange[0])),np.argmin(np.abs(wavelengths-emwavelrange[1])))


timeaverage=timeaverage.ravel()

test = timeaverage[emwavelindices[0]:emwavelindices[1]]
testwavelengths=wavelengths[emwavelindices[0]:emwavelindices[1]]

lormod = LorentzianModel(prefix='Lor_')   

pars = lormod.guess(test, x=testwavelengths)

constmod = ConstantModel(prefix='Const_') 
pars.update(constmod.make_params())

mod = lormod + constmod

init = mod.eval(pars, x=testwavelengths)
out = mod.fit(test, pars, x=testwavelengths)


plt.figure()
plt.plot(testwavelengths,test,label='experimental data')
# plt.plot(wavelengths, out.init_fit, 'k--', label='initial fit')
plt.plot(testwavelengths,out.best_fit,label='best fit')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Intensity')
plt.legend(loc=0)

print(out.fit_report())
fitreport=out.fit_report()

def lorentzian(x,amplitude,mu,sigma,background):
    lor = amplitude*2/(2*np.pi)*sigma/((x-mu)**2+sigma**2)+background
    return lor



amplitude = float(fitreport.split('amplitude:')[-1].split('+')[0])
# amplitudeerr = float(fitreport.split('amplitude:')[-1].split('+/-')[1].split('(')[0])
center = float(fitreport.split('center:')[-1].split('+')[0])
# centererr = float(fitreport.split('center:')[-1].split('+/-')[1].split('(')[0])
sigma = float(fitreport.split('sigma:')[-1].split('+')[0])
# sigmaerr = float(fitreport.split('sigma:')[-1].split('+/-')[1].split('(')[0])
background = 454
x=testwavelengths

lorplot= lorentzian(x,amplitude,center,sigma,background)
plt.figure()
plt.plot(x,lorplot)
plt.plot(wavelengths,timeaverage)

plt.figure()
plt.plot(x,lorplot)
plt.plot(testwavelengths,test)
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
meantemp = np.mean(Yfiltered,axis=0)
Yfilteredwav = np.zeros(Yfiltered.shape)
for j in range(len(Yfiltered[0])):
    Yfilteredwav[:,j] = Yfiltered[:,j]-meantemp[j]
    
if Debugmode==True:
    fig,ax=plt.subplots(2,1,sharex=True,sharey=False,constrained_layout=True) #plot excitation and emission above each other. If you want them to be visualized with the same window the yrange should cover the same distance. also zooming goes automatically
    emwavelrange=(595,630)
    emwavelindices=(np.argmin(np.abs(wavelengths-emwavelrange[0])),np.argmin(np.abs(wavelengths-emwavelrange[1])))
    timeminplot=0
    timemaxplot=-1
    vmax=2000
    vmax=np.max(Yfiltered[Yfiltered<np.max(Yfiltered)])
    # plt.figure()
    # plt.imshow(Yfiltered[emwavelindices[0]:emwavelindices[1],timeminplot:timemaxplot],extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths[emwavelindices[0]:emwavelindices[1]]),np.min(wavelengths[emwavelindices[0]:emwavelindices[1]])],aspect='auto')#,vmax=1000)
    # im0=ax[0].imshow(Yfiltered,extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths),np.min(wavelengths)],aspect='auto')
    im0=ax[0].imshow(Yfiltered[emwavelindices[0]:emwavelindices[1],:],extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(wavelengths[emwavelindices[0]:emwavelindices[1]]),np.min(wavelengths[emwavelindices[0]:emwavelindices[1]])],aspect='auto',vmin=0,vmax=vmax)
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
    # plt.tight_layout()
    
    
    fig,ax=plt.subplots(2,1,sharex=True,sharey=False,constrained_layout=True) #plot excitation and emission above each other. If you want them to be visualized with the same window the yrange should cover the same distance. also zooming goes automatically
    # emwavelrange=(600,640)
    emwavelindices=(np.argmin(np.abs(wavelengths-emwavelrange[0])),np.argmin(np.abs(wavelengths-emwavelrange[1])))
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
    im1=ax[1].imshow(excitationdata[0],extent=[0,len(Yfiltered[0]),np.max(excitationdata[1]),np.min(excitationdata[1][excitationdata[1]>0])],aspect='auto')
    # im1=ax[1].imshow(excitationdata[0],extent=[data[26][0]*dtmacro,data[26][-1]*dtmacro,np.max(excitationdata[1]),np.min(excitationdata[1][excitationdata[1]>0])])
    ax[1].set_ylabel('Excitation wavelength (nm)')
    ax[1].set_xlabel('Index')
    ax[1].set_title('Excitation')
    fig.colorbar(im1,ax=ax[1])
    ax[1].invert_yaxis()
   

#select excitation spectra based on trigger events from first photons in the emission cycle
#can do excitation emission correlation only on those. Forward and backward sweep combined

#%% excitation and emission correlation
# taus=[0,1,2,5,10,30,100]
# taus=[1,5,8,10,15]
# taus=[1,5,8,10,20]
taus=[0,1]
taus=[10,50]
# taus=[30,50]
# taus=[0,1,2]
# taus=[0]
# taus = [5,10,100,400]
# taus=np.arange(1,40,1)
# plot = 'cov'
plot = 'corr'
# plot = 'none'
# Emissiondata1 = Yfiltered[:,0:100] #can get postselection by selecting only a particular window. Note that you need emissiondata to plot insteada of Yfiltered
# Emissiondata2 = binnedintensities[:,0:100]
emwavelrange=(500,540)
emwavelrange=(600,640)
emwavelrange=(595,630)
# emwavelrange=(np.min(wavelengths),np.max(wavelengths))
emwavelindices=(np.argmin(np.abs(wavelengths-emwavelrange[0])),np.argmin(np.abs(wavelengths-emwavelrange[1])))

emissionwavelengths = wavelengths[emwavelindices[0]:emwavelindices[1]]


excwavelrange=(430,590)
excwavelrange=(500,580)
excwavelindices=(np.argmin(np.abs(excitationdata[1][:,1]-excwavelrange[0])),np.argmin(np.abs(excitationdata[1][:,1]-excwavelrange[1])))

excitationwavelengths = excitationdata[1][excwavelindices[0]:excwavelindices[1],1]

limlow = 200
limhigh = 500
Excitationdata = excitationdata[0][excwavelindices[0]:excwavelindices[1],limlow:limhigh]
Emissiondata = Yfiltered[emwavelindices[0]:emwavelindices[1],limlow:limhigh]-np.min(Yfiltered)


# emissioncovariance,emissionnormalization,emissioncorrelation = rplm.pearsoncorrelation(Excitationdata,Emissiondata,wavelengths,binnedwavelengths[:,0],taus=taus,plot=plot)
# select certain wavelengths pairs that were correlated in a line plot.
# most applicable to excitation and emission correlation

# excemmcovariance,excemmnormalization,excemmcorrelation = rplm.pearsoncorrelation(Emissiondata,Excitationdata,excitationwavelengths,emissionwavelengths,taus=taus,plot=plot)
# excemmcovariance,excemmnormalization,excemmcorrelation = rplm.spectralcorrelation(Emissiondata,Excitationdata,excitationwavelengths,emissionwavelengths,taus=taus,plot=plot)
# excitationcovariance,excitationnormalization,excitationcorrelation = rplm.pearsoncorrelation(Excitationdata,Excitationdata,excitationwavelengths,excitationwavelengths,taus=taus,plot=plot)
emissioncovariance,emissionnormalization,emissioncorrelation = rplm.pearsoncorrelation(Emissiondata,Emissiondata,emissionwavelengths,emissionwavelengths,taus=taus,plot=plot)

#%% tempp


emisspec = emissionspec
emisspecold=emisspec[0][0,:,:]
# emisphotons=(emisspecold-473*(2*ysize))*5.36/emisspec[3]['Gain']
emisphotons=(emisspecold-495)*5.36
plt.figure()
emwavelrange=(600,640)
emwavelindices=(np.argmin(np.abs(emisspec[1]-emwavelrange[0])),np.argmin(np.abs(emisspec[1]-emwavelrange[1])))
X, Y = np.meshgrid(data[26]*dtmacro, emisspec[1][emwavelindices[0]:emwavelindices[1]])
plt.pcolormesh(X, Y, emisphotons[emwavelindices[0]:emwavelindices[1],:len(data[26])])
# plt.imshow(emisphotons)
# plt.colorbar()

timebins=np.linspace(0,int(Texp),int(Texp*5))
wavelbins=np.linspace(500,595,190)
Histdata,xedges,yedges=np.histogram2d(data[3]*data[1],calibcoeffs[1]+InVoltagenew_c(data[19]*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],bins=[timebins,wavelbins])#,norm=mcolors.LogNorm())
X, Y = np.meshgrid(timebins, wavelbins)
plt.pcolormesh(X, Y, Histdata.T)
# plt.xlim(data[26][0]*dtmacro,data[26][-1]*dtmacro)
# plt.xlim(120,250)
plt.xlabel('Time (s)')
plt.ylabel('Excitation/Emission wavelength')

#%% correlation of fits of spectra
fitted=rplm.fitspectra(Emissiondata,emissionwavelengths,0,100,model='Lor',Debugmode=False)

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
# plt.figure()
g2=rplm.MakeG2(times0,times1,dtmicro,g2restime=dtmicro*4,nrbins=1000)

# plt.figure()
# plt.plot(g2[0],g2[1])

#%%
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
centerstart,scaling,length = 600,1/10,500
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
sigma = np.ones(len(mean))*1
# sigma = np.random.random(len(mean))*8

testwavelengths=np.linspace(595,605,400)

test2 = randomsamp(mean,amplitude,sigma,testwavelengths,nrphotons=2000,bins=600) 
# test2 = randomsamp2(mean,amplitude,sigma,testwavelengths) 
# test2 = randomsamp(mean,amplitude,sigma,2000) 

testplots=[1,10,50,500]
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


fitted=rplm.fitspectra(test2[2],test2[0],0,len(test2[2][0]),model='Gauss',Debugmode=False)
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

timeselectedmin=50
timeselectedmax=150
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


taus =  [1,2,4]
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
# covnormcorr = rplm.pearsoncorrelation(test2[2][minwavspec1:maxwavspec1,:timeend],test2[2][minwavspec1:maxwavspec1,:timeend],testwavelengths,testwavelengths,taus=taus,plot=plot)
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
    vmin=0
    vmax0corr=0.2
    # print(vmax0corr)
    plt.figure()
    # plt.imshow(emissioncorrelation[i],extent=[test2[0][minwavspec1],test2[0][maxwavspec1],test2[0][maxwavspec1],test2[0][minwavspec1]],vmin=vmin,vmax=vmax0corr)
    plt.imshow(emissioncorrelation[i],extent=[np.min(wavelengths),np.max(wavelengths),np.max(wavelengths),np.min(wavelengths)],vmin=vmin,vmax=vmax0corr)
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
plotrange=matchrange
plotrange=excwavelrange
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
binwidth=0.1
macrolimits = rpl.HistPhotons(macrotimesin*dtmacro,binwidth,Texp)

# macrotimesin=data[25]
# macrolimits=macrotimesin

limits=macrolimits
taulist=np.zeros(len(limits)-1)

taulist1=np.zeros(len(limits)-1)
taulist2=np.zeros(len(limits)-1)

#wavelbins=150
#Exspeclist=np.zeros([wavelbins,len(limits)-1])
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
#[taulist,Alist,ybglist] = Parallel(n_jobs=-1, max_nbytes=None)(delayed(processInput)(tbinnr) for tbinnr in tqdm(range(nrtbins-1)))
#plt.figure()
#test=np.zeros((wavelbins,len(limits)-1))
for tbinnr in tqdm(range(len(limits)-1)):
    microtimes = microtimesin[limits[tbinnr]:limits[tbinnr+1]]

    # [taulist[tbinnr],Alist[tbinnr],ybglist[tbinnr],buff[tbinnr]]=rpl.GetLifetime(microtimes,dtmicro,dtmacro,300e-9,tstart=lifetime[3]*dtmicro,histbinmultiplier=1,ybg=lifetime[2]*binwidth/Texp,plotbool=False,expterms=1,method='ML_c') 
    # lifetimetot=rpl.GetLifetime(microtimes,dtmicro,dtmacro,300e-9,tstart=lifetime[3]*dtmicro,histbinmultiplier=1,ybg=lifetime[2]*binwidth/Texp,plotbool=False,expterms=2,method='ML_c') 
    # [taulist1[tbinnr],taulist2[tbinnr],Alist1[tbinnr],Alist2[tbinnr],ybglist[tbinnr],buff[tbinnr]]=[lifetimetot[0][0],lifetimetot[0][1],lifetimetot[1][0],lifetimetot[1][1],lifetimetot[2],lifetimetot[3]]
    tauav[tbinnr]=(np.mean(microtimes)-lifetime[3])*dtmicro*1e9-1
    photonspbin[tbinnr]=len(microtimes)
    #using th
    #for when histogramming photons
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


#%% 2 detectors
macrotimesindet0=data[3]
macrotimesindet1=data[5]
microtimesindet0=data[2]
microtimesindet1=data[4]
binwidth=0.01
macrolimitsdet0 = rpl.HistPhotons(macrotimesindet0*dtmacro,binwidth,Texp)
macrolimitsdet1 = rpl.HistPhotons(macrotimesindet1*dtmacro,binwidth,Texp)
lengthlimits=int(Texp/binwidth)
# macrotimesin=data[25]
# macrolimits=macrotimesin
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
    ax1.plot(macrotimesindet0[limits[0:lengthlimits-1]]*dtmacro,photonspbintot,'b')
    #ax1.plot(macrotimesin[limits[0:lengthlimits-1]]*dtmacro,Alist,'b')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Photons per '+str(int(binwidth*1000))+' ms bin', color='b')
    #ax1.set_xlim([0,13])
    #ax1.set_ylim([0,0.2*np.max(ylistspec)])
    #ax1.ticklabel_format(style='sci',scilimits=(-2,3),axis='both')
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(macrotimesindet0[limits[0:lengthlimits-1]]*dtmacro,tauavtot,'r')
    #ax2.set_ylim([586,588])
    ax2.set_ylim([0,50])
    ax2.set_ylabel('Lifetime (ns)', color='r')
    ax2.tick_params('y', colors='r')


#%% Limits
limitex=400 #base these limits on the above-mentioned plot.
limittrlow=180
limittrhigh=240
limitoffhigh=100

lifetimeexciton=50
# lifetimetrionmin=4
# lifetimetrionmax=6
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

    if nphots>limitex and taulist[tbinnr]<lifetimeexciton:

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
            if microtimes[k]*dtmicro>7e-9 and microtimes[k]*dtmicro<14e-9:
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
fittrion=rpl.GetLifetime(microtimes_trion,dtmicro,dtmacro,5e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_trion/(Texp/binwidth),method='ML_c')
# fittrion=GetLifetime(microtimes_trion,dtmicro,dtmacro,5e-9,tstart=-1,plotbool=True,ybg=bins_trion*40*binwidth/np.max(microtimesblue),method='ML_c')

# fitoff=rpl.GetLifetime(microtimes_off,dtmicro,dtmacro,10e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_trion/(Texp/binwidth),method='ML_c')
fitmid=rpl.GetLifetime(microtimes_mid,dtmicro,dtmacro,200e-9,tstart=-1,plotbool=True,ybg=lifetime[2]*bins_mid/(Texp/binwidth),method='ML_c')

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
    

    
#%%
histsettings=(500,580,100)
plt.figure()
exspecex=rplm.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_ex*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
plt.plot(exspecex[0],exspecex[1]/np.sum(exspecex[1]),label='exciton')
exspectr=rplm.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_trion*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
plt.plot(exspectr[0],exspectr[1]/np.sum(exspectr[1]),label='trion')
plt.legend()
# exspec=rpl.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_mid*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
# plt.plot(exspec[0],(exspec[1]/np.sum(exspec[1])))
plt.xlabel('Excitation wavelength')
plt.ylabel('Intensity')
plt.title('Exciton and trion excitation spectra')

#%%temptry
histsettings=(500,580,100)
plt.figure()
exspecex=rplm.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_ex*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
plt.plot(exspecex[0],exspecex[1]/nrex)#/np.sum(exspecex[1]),label='exciton')
exspectr=rplm.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_trion*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
plt.plot(exspectr[0],exspectr[1]/nrtrion)#/np.sum(exspectr[1]),label='trion')
plt.legend()
# exspec=rpl.Easyhist(calibcoeffs[1]+rpl.InVoltagenew(macrotimescycle_mid*data[1],Freq,Vpp,Voffset,tshift)*calibcoeffs[0],histsettings[0],histsettings[1],histsettings[2])
# plt.plot(exspec[0],(exspec[1]/np.sum(exspec[1])))

plt.xlabel('Excitation wavelength')
plt.ylabel('Intensity')
plt.title('Exciton and trion excitation spectra')
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
t0=lifetime[3]*dtmicro*1e9
histsettings=(440,612,1)
# histsettings=(440,500,1)
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
plt.figure()
plt.plot(exspecB[0],(testA/interprefspec(exspecA[0])-testB/interprefspec(exspecB[0]))/(testB/interprefspec(exspecB[0])))
plt.plot(exspecB[0],np.zeros(len(exspecB[0])),'--k')
plt.ylim(-1,1)
# plt.plot(exspecB[0],(exspecB[1]-np.min(exspecB[1]))/np.sum(exspecB[1][20:100]-np.min(exspecB[1]))-(exspecA[1]-np.min(exspecA[1]))/np.sum(exspecA[1][20:100]-np.min(exspecA[1])))

#%% fit absorption curve
#at the moment it does not work and I do not know why. Based it a bit on the cubic background david wrote in his paper, but i can imagine that it is highly dependent on the starting positions
from lmfit.models import PolynomialModel
from lmfit import Model
# plt.figure()
# plt.plot(exspecex[0],exspecex[1]/interprefspec(exspecex[0]))
spectra=exspecex[1]/interprefspec(exspecex[0])
testwav=exspecex[0]
def func(x, a, b, c, d):
    return a + b * x + c * x ** 2 + d * x ** 3

lormod = LorentzianModel(prefix='Lor_')
pars = lormod.guess(spectra, x=testwav)
pars['Lor_center'].set(value=480)
lormod2=LorentzianModel(prefix='Lor2_')
pars.update(lormod2.make_params())
pars['Lor2_center'].set(value=430)
lormod3=LorentzianModel(prefix='Lor3_')
pars.update(lormod3.make_params())
pars['Lor3_center'].set(value=520)
lormod4=LorentzianModel(prefix='Lor4_')
pars.update(lormod4.make_params())
pars['Lor4_center'].set(value=560)
pmodel = Model(func)
pars.update(pmodel.make_params())

bkg = ConstantModel(prefix='contst')
pars.update(bkg.make_params())
bkg2 = LinearModel(prefix='lin')
pars.update(bkg2.make_params())

mod=lormod+lormod2+lormod3+lormod4+pmodel
init = mod.eval(pars, x=testwav)
out = mod.fit(spectra, pars, x=testwav)


plt.figure()
plt.plot(testwav,out.best_fit)
plt.plot(exspecex[0],exspecex[1]/interprefspec(exspecex[0]))



#%% Plot FLID map
plt.figure()
# plt.hist2d(tauav,photonspbin,(50,int(np.max(photonspbin))),range=[[0,50],[0,np.max(photonspbin)]],norm=mcolors.LogNorm())
plt.hist2d(taulist2,photonspbin,(50,int(np.max(photonspbin))),range=[[0,50],[0,np.max(photonspbin)]],norm=mcolors.LogNorm())
plt.title('FLID map')
plt.ylabel('Counts per bin')
plt.ylabel('Counts per '+str(int(binwidth*1000))+' ms bin')
plt.xlabel('Lifetime (ns)')
plt.colorbar()

#plt.plot(np.arange(0,50),np.arange(0,50)*505/27+8,'w')

#%% postselect on intensities and lifetimes
taumin=10
taumax=25
Imin=250
# Imin=150
Imax=400
# Imax=250

test=np.where((taumin<tauav)&(tauav<taumax)&(Imin<photonspbin)&(photonspbin<Imax))
# Ilist=np.where(Ilisttemp)
# plt.figure()
# plt.plot(tauav[test])
rawdata=np.asarray(test).ravel()
selecteddata=limits
originaldata=data[19]
wavelengthrange=matchrange
histogram=200

testdata=rplm.histogramexcitationspectra(Freq,Vpp,Voffset,tshift,calibcoeffs,dtmacro,rawdata,selecteddata,originaldata,wavelengthrange,histogram)

plt.figure()
plt.plot(testdata[1][:,1],np.sum(testdata[0],axis=1))
plt.title('Imin='+str(Imin)+'. Imax='+str(Imax))
plt.xlabel('Excitaiton wavelength')
plt.ylabel('Intensity')

# plt.imshow(testdata[0])
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
