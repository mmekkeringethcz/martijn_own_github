# -*- coding: utf-8 -*-
"""
Created on Fri May 29 14:57:02 2020

@author: rober
"""

import os, numpy as np, csv, matplotlib.pyplot as plt, scipy.optimize as opt, math, struct, binascii, gc, time, random
import multiprocessing
from operator import sub
from joblib import Parallel, delayed
import scipy#, lmfit
from scipy.optimize import minimize # used for implementation of maximum likelihood exponential fit
from matplotlib import gridspec
import matplotlib.colors as mcolors
from math import factorial
from math import *
from scipy.stats import poisson

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
# %matplotlib auto

# In[10]:


#I get errors unless the function that the cores execute in parallel is defined outside the class function 
def hist2(x,y,bins):
    store = np.zeros(len(bins)-1,dtype='float');
    for i in x:
        res = y.searchsorted(bins+i)
        store += res[1:]-res[:-1]
    return store
def load_obj(name, folder ):
    with open(folder + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# In[23]:



def ImportT3(filename,HHsettings={}):
    with open(filename, "rb+") as f:
    
        if filename[-3:]=='ptu':
        ##It is a .ptu file, information is in header
            while True:
                if f.read(1) == b'M':
                    if f.read(18) == b'easDesc_Resolution': # recognize time unit entry
                        break
            f.read(21) # rest of the tag
            dtmicro = struct.unpack('d',f.read(8))[0]
            #print('Microtime unit:', dtmicro)
        
            while True:
                if f.read(1) == b'M':
                    if f.read(24) == b'easDesc_GlobalResolution': # recognize time unit entry
                        break
            f.read(15) # rest of the tag
            dtmacro = struct.unpack('d',f.read(8))[0]
            #print('Macrotime unit:', dtmacro)
        
            while True:
                if f.read(1) == b'T':
                    if f.read(23) == b'TResult_NumberOfRecords': # recognize number of records entry
                        break
            f.read(16) # rest of the tag
            nrrec = struct.unpack('q',f.read(8))[0] # extract number of records
            #print('Number of records in file:', nrrec)
    
            while True:
                if f.read(1) == b'H':
                    if f.read(9) == b'eader_End':
                        #print('Header_End found')
                        break
            f.read(38) # rest of Header_End
        elif filename[-3:]=='out':
            ## it is an .out file and acquisition settings are in an extra file
            nrrec=HHsettings["overallCounts"]
            dtmicro=HHsettings["resolution"]*1e-12    #in s
            dtmacro=1/HHsettings["syncRate"]    #in s
            
            
        macrotimes0 = np.zeros(nrrec,dtype='int64'); #Macrotime of photons on detector 0 in integer units
        microtimes0 = np.zeros(nrrec,dtype='int64'); #Microtime of photons on detector 0 in integer units
        macrotimes1 = np.zeros(nrrec,dtype='int64');
        microtimes1 = np.zeros(nrrec,dtype='int64');
        macrotimesfireA = np.zeros(nrrec,dtype='int64');
        microtimesfireA = np.zeros(nrrec,dtype='int64');
        macrotimesfireB = np.zeros(nrrec,dtype='int64');
        microtimesfireB = np.zeros(nrrec,dtype='int64');
        macrotimesfireC = np.zeros(nrrec,dtype='int64');
        microtimesfireC = np.zeros(nrrec,dtype='int64');
        macrotimesfireD = np.zeros(nrrec,dtype='int64');
        microtimesfireD = np.zeros(nrrec,dtype='int64');
        macrotimesfireE = np.zeros(nrrec,dtype='int64');
        microtimesfireE = np.zeros(nrrec,dtype='int64');
        macrotimecycle0 = np.zeros(nrrec,dtype='int64'); #Time since start of current galvo cycle of photons on detector 0 in integer units (macrotime)
        macrotimeemissioncycle0 = np.zeros(nrrec,dtype='int64'); #Time since start of current emission cycle of photons on detector 0 in integer units (macrotime)
        macrotimecycle1 = np.zeros(nrrec,dtype='int64');
        cyclenumber0 = np.zeros(nrrec,dtype='int64'); #Number of galvo cycle of of photons on detector 0
        emissioncyclenumber0 = np.zeros(nrrec,dtype='int64'); #Number of emission cycle of of photons on detector 0
        firstphotonincycle0 = np.zeros(nrrec,dtype='int64'); #Index of the first photon on a given galvo cycle
        firstphotoninemissioncycle0 = np.zeros(nrrec,dtype='int64'); #Index of the first photon on a given emission cycle
        overflows = 0
        nrphotons0 = 0
        nrphotons1 = 0
        nrfireA = 0
        nrfireB = 0
        nrfireC = 0
        nrfireD = 0
        nrfireE = 0
        prevchann = 0
        lastcyclestarttime = 0
        lastemissioncyclestarttime = 0
        currentcycle = 0
        currentemissioncycle = 0
        
        for i in range(nrrec):
            entry = f.read(4)
            channel = struct.unpack("I",entry)[0] >> 25 # read channel number, first 7 bits
            if channel == 0:
                macrotime = (struct.unpack("I",entry)[0] & 0x3FF)
                macrotimes0[nrphotons0] = macrotime + 1024*overflows
                microtime = ((struct.unpack("I",entry)[0] >> 10) & 0x7FFF)
                microtimes0[nrphotons0] = microtime
                cyclenumber0[nrphotons0] = currentcycle
                macrotimecycle0[nrphotons0] = macrotime-lastcyclestarttime + 1024*overflows
                macrotimeemissioncycle0[nrphotons0] = macrotime-lastemissioncyclestarttime + 1024*overflows
                nrphotons0 += 1
                prevchann = 0
            elif channel == 1:
                macrotime = (struct.unpack("I",entry)[0] & 0x3FF)
                macrotimes1[nrphotons1] = macrotime + 1024*overflows
                microtime = ((struct.unpack("I",entry)[0] >> 10) & 0x7FFF)
                microtimes1[nrphotons1] = microtime
                macrotimecycle1[nrphotons1] = macrotime-lastcyclestarttime + 1024*overflows
                nrphotons1 += 1
                prevchann = 1
            elif channel == 127:
                nroverflows = (struct.unpack("I",entry)[0] & 0x3FF)
                overflows += nroverflows
                prevchann = 127
            elif channel == 65:
                macrotime = (struct.unpack("I",entry)[0] & 0x3FF)
                macrotimesfireA[nrfireA] = macrotime + 1024*overflows
                microtime = ((struct.unpack("I",entry)[0] >> 10) & 0x7FFF)
                microtimesfireA[nrfireA] = microtime
                nrfireA += 1
            elif channel == 66:
                macrotime = (struct.unpack("I",entry)[0] & 0x3FF)
                macrotimesfireB[nrfireB] = macrotime + 1024*overflows
                microtime = ((struct.unpack("I",entry)[0] >> 10) & 0x7FFF)
                microtimesfireB[nrfireB] = microtime
                nrfireB += 1  
            elif channel == 67:
                macrotime = (struct.unpack("I",entry)[0] & 0x3FF)
                macrotimesfireC[nrfireC] = macrotime + 1024*overflows
                microtime = ((struct.unpack("I",entry)[0] >> 10) & 0x7FFF)
                microtimesfireC[nrfireC] = microtime
                nrfireC += 1 
            elif channel == 68:
                macrotime = (struct.unpack("I",entry)[0] & 0x3FF)
                macrotimesfireD[nrfireD] = macrotime + 1024*overflows
                cyclenumber0[nrphotons0] = currentcycle
                currentcycle += 1
                firstphotonincycle0[currentcycle] = nrphotons0 + 1
                lastcyclestarttime = macrotime + 1024*overflows
                microtime = ((struct.unpack("I",entry)[0] >> 10) & 0x7FFF)
                microtimesfireD[nrfireD] = microtime
                nrfireD += 1
            elif channel == 72:
                macrotime = (struct.unpack("I",entry)[0] & 0x3FF)
                macrotimesfireE[nrfireE] = macrotime + 1024*overflows
                emissioncyclenumber0[nrphotons0] = currentemissioncycle
                firstphotoninemissioncycle0[currentemissioncycle] = nrphotons0 + 1
                currentemissioncycle += 1  
                lastemissioncyclestarttime = macrotime + 1024*overflows
                microtime = ((struct.unpack("I",entry)[0] >> 10) & 0x7FFF)
                microtimesfireE[nrfireE] = microtime
                nrfireE += 1                              
            else:
                print('bad channel:',channel)
                
    microtimes0 = microtimes0[:nrphotons0]
    macrotimes0 = macrotimes0[:nrphotons0]
    cyclenumber0 = cyclenumber0[:nrphotons0]
    emissioncyclenumber0 = emissioncyclenumber0[:nrphotons0]
    firstphotonincycle0 = firstphotonincycle0[:currentcycle]
    firstphotoninemissioncycle0 = firstphotoninemissioncycle0[:currentemissioncycle]
    macrotimecycle0 = macrotimecycle0[:nrphotons0]
    macrotimeemissioncycle0 = macrotimeemissioncycle0[:nrphotons0]
    microtimes1 = microtimes1[:nrphotons1]
    macrotimes1 = macrotimes1[:nrphotons1]
    microtimesfireA = microtimesfireA[:nrfireA]
    macrotimesfireA = macrotimesfireA[:nrfireA]
    microtimesfireB = microtimesfireB[:nrfireB]
    macrotimesfireB = macrotimesfireB[:nrfireB]
    microtimesfireC = microtimesfireC[:nrfireC]
    macrotimesfireC = macrotimesfireC[:nrfireC]
    microtimesfireD = microtimesfireD[:nrfireD]
    macrotimesfireD = macrotimesfireD[:nrfireD]
    microtimesfireE = microtimesfireE[:nrfireE]
    macrotimesfireE = macrotimesfireE[:nrfireE]
    
    print('nrphotons0:',nrphotons0)
    print('nrphotons1:',nrphotons1)
    print('nrfireA:',nrfireA)
    print('nrfireB:',nrfireB)
    print('overflows:',overflows)
    # print('firstphotonincycle0',firstphotonincycle0)
    # print('firstphotoninemissioncycle0',firstphotoninemissioncycle0)
    

    return [dtmicro, dtmacro, microtimes0, macrotimes0, microtimes1, macrotimes1, nrphotons0,nrphotons1,overflows,microtimesfireA,macrotimesfireA,nrfireA,microtimesfireB,macrotimesfireB,nrfireB,macrotimesfireC,nrfireC,macrotimesfireD,nrfireD,macrotimecycle0,macrotimecycle1,cyclenumber0,firstphotonincycle0, macrotimeemissioncycle0, emissioncyclenumber0, firstphotoninemissioncycle0, macrotimesfireE, microtimesfireE]
#           0      , 1      , 2          , 3          , 4          , 5          , 6         , 7        , 8       , 9             , 10             , 11   , 12            , 13            , 14    , 15            , 16    , 17            , 18    , 19            , 20            , 21          , 22               ,  23                         24                 ,25


def ShiftPulsedData(microtimes0,microtimes1,macrotimes0,macrotimes1,dtmicro,dtmacro):
    dtmax = 8
    
    [ylist1,xlist1] = np.histogram(microtimes1,int(dtmacro/dtmicro),[0,int(dtmacro/dtmicro)])
    [ylist0,xlist0] = np.histogram(microtimes0,int(dtmacro/dtmicro),[0,int(dtmacro/dtmicro)])
    tlist = (xlist0[:-1]+0.5*(xlist0[1]-xlist0[0]))*dtmicro*1e9

    corrx = []; corry = [] # find shift for which the two decay curves overlap most
    for i in range(-dtmax,dtmax):
        corrx.append(i)
        corry.append(sum(ylist1[dtmax:-dtmax]*ylist0[dtmax+i:-dtmax+i]))
    xmax = corry.index(max(corry))
    shift = corrx[xmax]
    
    tlist0 = (microtimes0-shift) + macrotimes0*int(dtmacro/dtmicro)
    tlist1 = microtimes1 + macrotimes1*int(dtmacro/dtmicro) #in units of dtmicro
    
#    plt.xlabel('time (ns)')
#    plt.ylabel('counts (a.u.)')
#    p1, = plt.plot(tlist,ylist0+ylist1)
#    p2, = plt.plot(tlist,ylist0)
#    p3, = plt.plot(tlist,ylist1)
#    plt.([p1,p2,p3], ["APD0 + APD1","APD0","APD1"])
           
    return(microtimes0-shift,microtimes1,tlist0,tlist1,dtmicro,dtmacro,tlist,ylist0+ylist1)

def GetLifetime(microtimes,dtmicro,dtmacro,dtfit,tstart=-1,histbinmultiplier=1,ybg=0,plotbool=False,method='ML',expterms=1): 
    # microtimes = microtimes array with photon events
    # dtfit is the time interval considered for the fit [s], tstart [s] is the starting point of the fit within the histogram. If set to -1 it starts at the time with the highest intensity.
    # histbinmultiplier is a multiplier. actual binwidth is given as histbinmultiplier*dtmicro[s]
    # ybg is the background considered for the fit (CHECK UNITS!!). If set to -1 --> try to estimate background based on last bins. set to 0 --> no background subtraction
    # plotbool: plot histogram with fit
#    print('Chosen method is:' + method)
    [ylist,xlist] = np.histogram(microtimes,int(dtmacro/(dtmicro*histbinmultiplier)),[0,int(dtmacro/dtmicro)])
    tlist = (xlist[:-1]+0.5*(xlist[1]-xlist[0]))*dtmicro*1e9
#    print(histbinmultiplier)
    istart = int(tstart/dtmicro) #find index of maximum element in ylist
    if istart < 0:
        istart = ylist.argmax()
    iend = istart + int(dtfit/(dtmicro*histbinmultiplier))
    if iend>len(tlist):
        iend = len(tlist) 
        
    # get background (by simply looking at last ten data points) and substract from intensity data.
    if ybg < 0:
        ybg = np.mean(ylist[-100:]) # mean background per histogram bin bin of length
   
    if method == 'ML': #maximum likelihood exponential fit
        [tau1fit,A1fit] = MaxLikelihoodFit(tlist,ylist,istart,iend,ybg,False,expterms)
    elif method == 'ML_c': #maximum likelihood exponential fit
        [tau1fit,A1fit] = MaxLikelihoodFit_c(tlist,ylist,istart,iend,ybg,False,expterms)
    elif method == 'WLS': # weighted least squares fit
        [taufit,Afit] = WeightedLeastSquareFit(tlist,ylist,istart,iend,ybg,plotbool=False)
    else:
        taufit = 0; Afit = 0;
        print('Error: invalid fit method')
    
    if plotbool == True:
        plt.xlabel('time (ns)')
        plt.ylabel('')
        plt.semilogy(tlist,ylist,'.',tlist[istart:iend],np.sum(np.array([A1fit[k]*np.exp(-(tlist[istart:iend]-tlist[istart])/tau1fit[k])+ybg*(k<1) for k in range(expterms)]),0))
        plt.semilogy([tlist[0],tlist[-1]],[ybg,ybg],'k--')
        plt.show()
        print('Fitted lifetime:',tau1fit,'ns; Amax:',A1fit)


    # if plotbool == True:
    #     # yest = np.array([Aest[k]*np.exp(-(xdata[i]-xdata[0])/tauest[k])+bgcpb for i in range(len(xdata))])
    #     yest = np.array([np.sum([A1fit[k]*np.exp(-(tlist[istart:iend]-tlist[istart])/tau1fit[k])+ybg*(k<1) for k in range(expterms)]) for i in range(len(xdata))])
        
    #     plt.figure()
    #     plt.semilogy(tlist,ylist,'.',xdata,yest,[xdata[1],xdata[-1]],[bgcpb,bgcpb],'k--')
    #     # plt.xlim([xdata[1],xdata[-1]])
    #     plt.show()   
    # Amax is the maximum y-value
    Amax = np.max(ylist)
        
    return(tau1fit,A1fit,ybg,istart)  

def HistPhotons(photontlist,binwidth,Texp): # finds the index of the first photon in each bin. photontlist is in [s]
    histmax = Texp # experiment duration [s]

    nrbins = int(Texp/binwidth)
    limits = np.full(nrbins,len(photontlist))
    counter,i = 0,0
    while counter < nrbins and i < len(photontlist):
        while photontlist[i] > counter*binwidth:
            limits[counter] = i
            counter += 1
        i += 1
    
    return(limits)

def MakeIntTrace(limits0,limits1,binwidth,Texp):
    nrbins = int(Texp/binwidth)
    inttrace = np.array([(limits0[binnr+1]-limits0[binnr]+limits1[binnr+1]-limits1[binnr]) for binnr in range(nrbins-1)])

    fig = plt.figure(figsize=(15,3))
    gs = gridspec.GridSpec(1,2,width_ratios=[4,1])
    ax0 = plt.subplot(gs[0])
    p0 = ax0.plot(np.arange(len(inttrace))*binwidth,inttrace,'-',linewidth=0.5)
    plt.xlabel('time (s)')
    plt.ylabel('counts / %i ms' %(binwidth*1e3))
    plt.xlim([0,Texp])
    plt.ylim([0,1.1*np.max(inttrace)])

    histogram = np.histogram(inttrace,max(inttrace),[0,max(inttrace)])
    
    ax1 = plt.subplot(gs[1])
    ax1.plot(histogram[0],0.5*(histogram[1][:-1]+histogram[1][1:]))
    plt.xlabel('occurrence')
    plt.ylabel('counts / %i ms' %(binwidth*1e3))
    plt.ylim([0,1.1*np.max(inttrace)])
    
    return(inttrace)
    
    
#def MakeExTrace(limits0,binwidth,Texp):
#    nrbins = int(Texp/binwidth)
#    inttrace = np.array([(limits0[binnr+1]-limits0[binnr]+limits1[binnr+1]-limits1[binnr]) for binnr in range(nrbins-1)])
#
#    fig = plt.figure(figsize=(15,3))
#    gs = gridspec.GridSpec(1,2,width_ratios=[4,1])
#    ax0 = plt.subplot(gs[0])
#    p0 = ax0.plot(np.arange(len(inttrace))*binwidth,inttrace,'-',linewidth=0.5)
#    plt.xlabel('time (s)')
#    plt.ylabel('counts / %i ms' %(binwidth*1e3))
#    plt.xlim([0,Texp])
#    plt.ylim([0,1.1*np.max(inttrace)])
#
#    histogram = np.histogram(inttrace,max(inttrace),[0,max(inttrace)])
#    
#    ax1 = plt.subplot(gs[1])
#    ax1.plot(histogram[0],0.5*(histogram[1][:-1]+histogram[1][1:]))
#    plt.xlabel('occurrence')
#    plt.ylabel('counts / %i ms' %(binwidth*1e3))
#    plt.ylim([0,1.1*np.max(inttrace)])
#    
#    return(inttrace)

def MakeTauTrace(taubinlist,intbinlist,binwidth,Texp,taumin=0,taumax=100,intmin=0,intmax=100,col='k'):
    nrbins = int(Texp/binwidth)
  
    fig = plt.figure(figsize=(15,7))
    gs = gridspec.GridSpec(2,2,width_ratios=[4,1])
    ax0 = plt.subplot(gs[0])
    p0 = ax0.plot(np.arange(len(intbinlist))*binwidth,intbinlist,'-',linewidth=0.5,color=col)
    plt.xlabel('time (s)')
    plt.ylabel('counts / %i ms' %(binwidth*1e3))
    plt.xlim([0,Texp])
    plt.ylim([intmin,intmax])

    histogram = np.histogram(intbinlist,int(np.max(intbinlist)),[0,int(np.max(intbinlist))])
    
    ax1 = plt.subplot(gs[1])
    ax1.plot(histogram[0],0.5*(histogram[1][:-1]+histogram[1][1:]),color=col)
    plt.xlabel('occurrence')
    plt.ylabel('counts / %i ms' %(binwidth*1e3))
    plt.ylim([intmin,intmax])
    
    ax2 = plt.subplot(gs[2])
    p2 = ax2.plot(np.arange(len(intbinlist))*binwidth,taubinlist,'.',markersize=1.5,color=col)
    plt.xlabel('time (s)')
    plt.ylabel('lifetime (ns)')
    plt.xlim([0,Texp])
    plt.ylim([taumin,taumax])

    histogram = np.histogram(taubinlist,taumax-taumin,[taumin,taumax])
    
    ax3 = plt.subplot(gs[3])
    ax3.plot(histogram[0],0.5*(histogram[1][:-1]+histogram[1][1:]),color=col)
    plt.xlabel('occurrence')
    plt.ylabel('lifetime (ns)')
    plt.ylim([taumin,taumax])
    
    plt.hold(True)
    

def BinIntensity(microtimes0,times0,limits0,microtimes1,times1,limits1,dtmicro,dtmacro,onintlim,offintlim):
    ## select only data with high or low intensity

    plt.title('total decay')
    tauave = GetLifetime(np.append(microtimes0,microtimes1),dtmicro,dtmacro,200e-9,-1)

    nrbins = len(limits0)
    inttrace = np.array([limits0[binnr+1]-limits0[binnr]+limits1[binnr+1]-limits1[binnr] for binnr in range(nrbins-1)])

    # find photons in on period
    onphotonlist0 = np.array([np.arange(limits0[binnr],limits0[binnr+1]) for binnr in range(nrbins-1) if inttrace[binnr] >= onintlim])
    onphotonlist0 = np.concatenate(onphotonlist0).ravel()
    onphotonlist1 = np.array([np.arange(limits1[binnr],limits1[binnr+1]) for binnr in range(nrbins-1) if inttrace[binnr] >= onintlim])
    onphotonlist1 = np.concatenate(onphotonlist1).ravel()

    onmicrotimes0 = np.array([microtimes0[i] for i in onphotonlist0])
    onmicrotimes1 = np.array([microtimes1[i] for i in onphotonlist1])
    ontimes0 = np.array([times0[i] for i in onphotonlist0])
    ontimes1 = np.array([times1[i] for i in onphotonlist1])
    plt.title('on decay')
    ontauave = GetLifetime(np.append(onmicrotimes0,onmicrotimes1),dtmicro,dtmacro,200e-9,-1)

    # find photons in off period
    offphotonlist0 = np.array([np.arange(limits0[binnr],limits0[binnr+1]) for binnr in range(nrbins-1) if inttrace[binnr] < offintlim])
    offphotonlist0 = np.concatenate(offphotonlist0).ravel()
    offphotonlist1 = np.array([np.arange(limits1[binnr],limits1[binnr+1]) for binnr in range(nrbins-1) if inttrace[binnr] < offintlim])
    offphotonlist1 = np.concatenate(offphotonlist1).ravel()

    offmicrotimes0 = np.array([microtimes0[i] for i in offphotonlist0])
    offmicrotimes1 = np.array([microtimes1[i] for i in offphotonlist1])
    offtimes0 = np.array([times0[i] for i in offphotonlist0])
    offtimes1 = np.array([times1[i] for i in offphotonlist1])
    plt.title('off decay')
    offtauave = GetLifetime(np.append(offmicrotimes0,offmicrotimes1),dtmicro,dtmacro,10e-9,-1)
    
    return(onmicrotimes0,offmicrotimes0,ontimes0,offtimes0,onmicrotimes1,offmicrotimes1,ontimes1,offtimes1)

def SliceHistogram(microtimes0,times0,limits0,macrotimescycle0,microtimes1,times1,limits1,macrotimescycle1,dtmicro,dtmacro,Imin,Imax):
    ## select only data with intensity between Imin and Imax

    nrbins = len(limits0)
    inttrace = np.array([limits0[binnr+1]-limits0[binnr]+limits1[binnr+1]-limits1[binnr] for binnr in range(nrbins-1)])

    # find photons in bins with intensities in (Imin,Imax] range
    onphotonlist0 = np.array([np.arange(limits0[binnr],limits0[binnr+1]) for binnr in range(nrbins-1) if Imin < inttrace[binnr] <= Imax])
    onphotonlist0 = np.concatenate(onphotonlist0).ravel()
    onphotonlist1 = np.array([np.arange(limits1[binnr],limits1[binnr+1]) for binnr in range(nrbins-1) if Imin < inttrace[binnr] <= Imax])
    onphotonlist1 = np.concatenate(onphotonlist1).ravel()
      
    onmicrotimes0 = np.array([microtimes0[i] for i in onphotonlist0])
    onmicrotimes1 = np.array([microtimes1[i] for i in onphotonlist1])
    ontimes0 = np.array([times0[i] for i in onphotonlist0])
    ontimes1 = np.array([times1[i] for i in onphotonlist1])
    onmacrotimescycle0 = np.array([macrotimescycle0[i] for i in onphotonlist0])
    onmacrotimescycle1 = np.array([macrotimescycle1[i] for i in onphotonlist0])
    
    # count nr of time bins with intensities corresponding to slice intensity
    onbincount = 0
    for binnr in range(nrbins-1):
        if Imin < inttrace[binnr] <= Imax:
            onbincount +=1

    return(onmicrotimes0,ontimes0,onmacrotimescycle0,onmicrotimes1,ontimes1,onmacrotimescycle1,onbincount)



def MaxLikelihoodFit(tlist,ylist,istart,iend,bgcpb,plotbool=False):
    ### Maximum likelihood routine to fit single exponential. Pro: Works also for small amount of data (single bins of 10ms!)
    # tlist: x-axis values, here time in ns; ylist: y-axis values, here cts per tlist-bin; istart and iend: first and last element of tlist and ylist that are considered for the fit.

    # check if istart and iend are good numbers
    if istart<0 or istart>=len(ylist):
        istart = 0
        print('WARNING: adapted istart in MaxLikelihoodExpFit')
    if iend<=istart or iend>len(ylist):
        iend = len(ylist)
        print('WARNING: adapted iend in MaxLikelihoodExpFit')

    # shift t0 to t=0
    ydata = ylist[istart:iend]
    xdata = tlist[istart:iend]

    # do calculations
    initParams = [np.max(ydata), 25] #initial guess for A and tau
    results = minimize(MaxLikelihoodFunction, initParams, args=(xdata,ydata,bgcpb),method='Nelder-Mead') # minimize the negative of the maxlikelihood function instead of maximimizing
    A1est = results.x[0] # get results of fit, A
    tau1est = results.x[1] # get results of fit, tau
#    A2est = results.x[2] # get results of fit, A
#    tau2est = results.x[3] # get results of fit, tau

#    if plotbool == True:
#        yest = np.array([A1est*np.exp(-(xdata[i]-xdata[0])/tau1est)+A2est*np.exp(-(xdata[i]-xdata[0])/tau2est)+bgcpb for i in range(len(xdata))])
#        plt.semilogy(tlist,ylist,'.',xdata,yest,[xdata[1],xdata[-1]],[bgcpb,bgcpb],'k--')
#        plt.show()        
    return(tau1est,A1est)#,tau2est,A2est)
    
def MaxLikelihoodFit_c(tlist,ylist,istart,iend,bgcpb,plotbool=False,expterms=1):
    ### Maximum likelihood routine to fit single exponential. Pro: Works also for small amount of data (single bins of 10ms!)
    # tlist: x-axis values, here time in ns; ylist: y-axis values, here cts per tlist-bin; istart and iend: first and last element of tlist and ylist that are considered for the fit.

    # check if istart and iend are good numbers
    if istart<0 or istart>=len(ylist):
        istart = 0
        print('WARNING: adapted istart in MaxLikelihoodExpFit')
    if iend<=istart or iend>len(ylist):
        iend = len(ylist)
        print('WARNING: adapted iend in MaxLikelihoodExpFit')

    # shift t0 to t=0
    ydata = ylist[istart:iend]
    xdata = tlist[istart:iend]

    # do calculations
    initParams = [np.max(ydata)*np.ones(expterms), 25*np.ones(expterms)] #initial guess for A and tau
    if expterms>1:
        initParams[1][0]=initParams[1][1]/10 #Make first component fast
    if expterms>2:
        initParams[1][2]=initParams[1][1]/5 #Make third component slow
    initParams = np.concatenate(initParams).ravel().tolist()
    results = minimize(MaxLikelihoodFunction_c, initParams, args=(xdata,ydata,bgcpb,expterms),method='Nelder-Mead') # minimize the negative of the maxlikelihood function instead of maximimizing
    Aest = results.x[0:expterms] # get results of fit, A
    tauest = results.x[expterms:2*expterms] # get results of fit, tau

#    if plotbool == True:
#        yest = np.array([Aest*np.exp(-(xdata[i]-xdata[0])/tauest)+bgcpb for i in range(len(xdata))])
#        plt.semilogy(tlist,ylist,'.',xdata,yest,[xdata[1],xdata[-1]],[bgcpb,bgcpb],'k--')
#        plt.show()        


    if plotbool == True:
        # yest = np.array([Aest[k]*np.exp(-(xdata[i]-xdata[0])/tauest[k])+bgcpb for i in range(len(xdata))])
        yest = np.array([np.sum([Aest[k]*np.exp(-(xdata[i]-xdata[0])/tauest[k])+bgcpb*(k<1) for k in range(expterms)]) for i in range(len(xdata))])
        plt.figure()
        plt.plot(tlist,ylist,'.',xdata,yest,[xdata[1],xdata[-1]],[bgcpb,bgcpb],'k--')
        plt.xlim([xdata[1],xdata[-1]])
        plt.show()        
        
    return(tauest,Aest)

def MaxLikelihoodFunction(params,xdata,ydata,const,expterms): 
    # max likelihood function for A*exp(-t/tau), needed in function MakLikelihoodFit
    # params = [A,tau]
    A = params[0:expterms]
    tau = params[expterms:2*expterms]
    model = const*np.ones(len(xdata))
    for k in range(len(A)):
        # print(A[k])
        # print(xdata)
        model+=A[k]*np.exp(-(xdata-xdata[0])/tau[k])
    # # model = A1*np.exp(-(xdata-xdata[0])/tau1)+const
    model [model <= 0] = 1e-10
# #    A2 = params[2]
# #    tau2 = params[2]
    E = 0;
    for i in range(len(xdata)):
# # #        E = E + ydata[i]*np.log(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+A2*np.exp(-(xdata[i]-xdata[0])/tau2)+const)-(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+A2*np.exp(-(xdata[i]-xdata[0])/tau2)+const)
# #         # E = E + ydata[i]*np.log(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+const)-(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+const)
        E = E + ydata[i]*np.log(model[i])-(model[i])
    return(-E) # This function needs to be MINIMIZED (because of the minus sign) to have the maximum likelihood fit!
MaxLikelihoodFunction_c = nb.jit(nopython=True)(MaxLikelihoodFunction)
def MaxLikelihoodFunction_Biexp(params,xdata,ydata,const): 
    # max likelihood function for A*exp(-t/tau), needed in function MakLikelihoodFit
    # params = [A,tau]
    A1 = params[0][0]
    tau1 = params[1][0]  
    A2 = params[0][1]
    tau2 = params[4] 
    model = A1*np.exp(-(xdata-xdata[0])/tau1)+A2*np.exp(-(xdata-xdata[0])/tau2)+const
    model [model <= 0] = 1e-10
#    A2 = params[2]
#    tau2 = params[2]
    E = 0;
    for i in range(len(xdata)):
#        E = E + ydata[i]*np.log(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+A2*np.exp(-(xdata[i]-xdata[0])/tau2)+const)-(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+A2*np.exp(-(xdata[i]-xdata[0])/tau2)+const)
        # E = E + ydata[i]*np.log(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+const)-(A1*np.exp(-(xdata[i]-xdata[0])/tau1)+const)
        E = E + ydata[i]*np.log(model[i])-(model[i])
    return(-E) # This function needs to be MINIMIZED (because of the minus sign) to have the maximum likelihood fit!


def processInput(tbinnr):
    microtimes = np.append(microtimesin[limits[tbinnr]:limits[tbinnr+1]])
    [ylist,xlist] = np.histogram(microtimes,int(dtmacro/(dtmicro*histbinmultiplier)),[0,int(dtmacro/dtmicro)])    
    tlist = (xlist[:-1]+0.5*(xlist[1]-xlist[0]))*dtmicro*1e9 # convert x-axis to time in ns
#    plt.clf()
#    plt11 = plt.figure(11)
    initParams = [np.max(ylist), 25]
    result=MaxLikelihoodFit_c(tlist,ylist,istart,iend,bgcpb,initParams,plotbool=False)
#    Ampl[tbinnr]=result[0]
#    Alpha[tbinnr]=result[1]
#    Theta[tbinnr]=result[2]
    return(result)


def InVoltage(t,Freq,VPP,VOffset,Verror):
    Period=1/Freq
    Bool1=t<=Period/4
    Bool2=np.logical_and(t>Period/4,t<=3*Period/4)
    Bool3=t>3*Period/4
    InVoltage=(VPP/2*t/(Period/4)+VOffset-Verror)*Bool1+(VPP/2-VPP/2*(t-Period/4)/(Period/4)+VOffset+Verror)*Bool2+(-VPP/2+VPP/2*(t-3*Period/4)/(Period/4)+VOffset-Verror)*Bool3
    return(InVoltage)

def InVoltagenew(t,Freq,VPP,VOffset,tshift):
    Period=1/Freq
    t=t+tshift
    Bool1=t<=Period/4
    Bool2=np.logical_and(t>Period/4,t<=3*Period/4)
    Bool3=t>3*Period/4
    InVoltage=(VPP/2*t/(Period/4)+VOffset)*Bool1+(VPP/2-VPP/2*(t-Period/4)/(Period/4)+VOffset)*Bool2+(-VPP/2+VPP/2*(t-3*Period/4)/(Period/4)+VOffset)*Bool3
    return(InVoltage)

def importASC(filename,nr_px=1024):
    """Import ASC files exported from Andor Solis.
    
    Parameters
    ----------
    filename: String with the complete path to the file.

    Returns
    -------
    raw_data : Video: 2D numpy array containing the counts [yPixels,xPixels], Video: 3D numpy array containing the counts [yPixels,xPixels,Frames]
    x: horizontal pixel number (image) or corresponding wavelength (spectrum), depending on input file type [Nx1]
    y: vertical pixel number [Nx1]
    info: dictionary with additional information such as exposure time

    Notes
    -----
    Export the file from andor as comma (,) separated file with appended aquisition information.

    References
    ----------
    

    Examples
    --------
    >>> [data,x,y,info] = importASC('X:/temp/LAB_DATA/Andor Spectrometer/Felipe/file.asc')
    
    """
    a = pd.read_csv(filename, header = None, low_memory = False)
    
# reffolder+'HR63_10MHz_200Hz_240mVpp_off0mV_OD1_cryo_camera.asc'
    
    info = {}
    #check length of series
    if a[0].str.contains('Number in Series').any()==True: # so in some datas this thing is called .. Series and otherwise ..Kinetic series
        buff = a.iloc[:,0].str.find('Number in Series',0)    #This one does not appear when using the whole image
        if pd.Series.max(buff)<0:       #Didn't appear. Whole image was used
            info['noFrames']=1
        elif pd.Series.max(buff)>=0:       #Cropped
            info['noFrames']=int(a.iloc[buff[buff==0].index[0],0].split(":")[1])
            #Data structure is: ["left","right",bottom","top"]
    elif a[0].str.contains('Number in Kinetics Series').any()==True:
        buff = a.iloc[:,0].str.find('Number in Kinetics Series',0)    #This one does not appear when using the whole image
        if pd.Series.max(buff)<0:       #Didn't appear. Whole image was used
            info['noFrames']=1
        elif pd.Series.max(buff)>=0:       #Cropped
            info['noFrames']=int(a.iloc[buff[buff==0].index[0],0].split(":")[1])
            #Data structure is: ["left","right",bottom","top"]
    else:
        info['noFrames']=1
   
    buff=a.iloc[:,0].str.find('Exposure Time',0)   #Gives number of column where it finds the string and -1 if not found
    info['expTime']=float(a.iloc[buff[buff==0].index[0],0].split(":")[1])
    buff=a.iloc[:,0].str.find('Gain level',0)   #Gives number of column where it finds the string and -1 if not found
    info['Gain']=int(a.iloc[buff[buff>-1].index[0],0].split(":")[1])   
    
    buff=a.iloc[:,0].str.find('Readout Mode',0)
    info['ReadMode']=(a.iloc[buff[buff==0].index[0],0].split(":")[1])
    if a.iloc[buff[buff==0].index[0],0].find('Image')>-1:
        # buff=a.iloc[:,0].str.find('Horizontal Binning',0)   #Gives number of column where it finds the string and -1 if not found
        # info['Horbinning']=int(a.iloc[buff[buff==0].index[0],0].split(":")[1])
        
        # buff=a.iloc[:,0].str.find('Vertical Binning',0)   #Gives number of column where it finds the string and -1 if not found
        # info['Vertbinning']=int(a.iloc[buff[buff==0].index[0],0].split(":")[1])
        
        #Find cropping
        buff = a.iloc[:,0].str.find('left',0)    #This one does not appear when using the whole image
        if pd.Series.max(buff)<0:       #Didn't appear. Whole image was used
            info['cropped']=False
            info['nrP_x'] = 1024
            info['nrP_y'] = 1024
        elif pd.Series.max(buff)>=0:       #Cropped
            info['cropped']=True
            info['Imlimits']=[int(a.iloc[buff[buff>=0].index[0],3].split(":")[1]),int(a.iloc[buff[buff>=0].index[0],4]),int(a.iloc[buff[buff>=0].index[0],5]),int(a.iloc[buff[buff>=0].index[0],6])]
            info['nrP_x'] = info['Imlimits'][1]-info['Imlimits'][0]+1
            info['nrP_y'] = info['Imlimits'][3]-info['Imlimits'][2]+1
        #Data structure is: ["left","right",bottom","top"]

   
        buff=a.iloc[:,0].str.find('Vertical Binning',0)
        if pd.Series.max(buff)<0:
            info['vertBin']=1
        else:
            info['vertBin']=int(a.iloc[buff[buff==0].index[0],0].split(":")[1])
            info['nrP_y']=int(info['nrP_y']/info['vertBin'])
            
        buff=a.iloc[:,0].str.find('Horizontal Binning',0)
        if pd.Series.max(buff)<0:
            info['horzBin']=1
        else:
            info['horzBin']=int(a.iloc[buff[buff==0].index[0],0].split(":")[1])
            info['nrP_x']=int(info['nrP_x']/info['horzBin'])
    elif a.iloc[buff[buff==0].index[0],0].find('Multi-Track')>-1:
        info['nrP_x']=1024
        info['nrP_y']=1
    
    x = a.iloc[0:(info['nrP_x']),0].str.strip().values.astype(float)
    y = np.arange(1,info['nrP_y']+1)
    
    raw_data = np.transpose(np.reshape(np.transpose(a.iloc[0:info['nrP_x']*info['noFrames'],1:-1].values.astype(int)),(info['nrP_y'],info['noFrames'],info['nrP_x'])),(0,2,1))
    if info['noFrames']==1:
        raw_data=raw_data[:,:,0]
    #Has dimensions ("yaxis","xaxis","numberofFrame")
    return [raw_data, x, y, info] 
def importPLE(filename):
    a = pd.read_csv(filename, delimiter=';', header = 20, low_memory = False)
    datalength=a.shape[0]
    wavelength=a.iloc[0:datalength,0].values
    intensity=a.iloc[0:datalength,1].values
    return [wavelength,intensity]



def MakeG2(times0,times1,dtmicro,g2restime=8e-9,nrbins=200):
    i0=0
    i1=0
    lim1=0
    g2 = np.zeros(2*nrbins)
    #g2B = np.zeros(2*nrbins)
    #g2C = np.zeros(2*nrbins)
    #blindB = 2e-9
    #blindC = 5e-9

    g2res = g2restime/dtmicro #transform g2restime [s] to g2res [microtime units]
    #blindB = blindB/tmicro
    #blindC = blindC/tmicro

    # correlate det0 with det1 (positive time differences)
    for i0 in range(len(times0)):
        t0 = times0[i0]
        i1 = 0
        q = 0
        while q == 0: 
            if lim1 + i1 < len(times1): # check if we've already reached end of photon stream on det1
                dt = times1[lim1+i1]-t0 
                if dt < 0: # find index lim1 of first photon on det1 that came after photon i0 on det0
                    lim1 = lim1 + 1
                else:
                    binnr = int(dt/g2res) # calculate binnr that corresponds to dt
                    if binnr < nrbins: # check if time difference is already large enough to stop correlation
                        g2[nrbins + binnr] += 1 # increase counter in corresponding bin by one
                        #if microtimes0[i0] > blindB and microtimes1[lim1+i1] > blindB:poi
                            #g2B[nrbins + binnr] += 1
                        #if microtimes0[i0] > blindC and microtimes1[lim1+i1] > blindC:
                            #g2C[nrbins + binnr] += 1
                        i1 = i1 + 1 # look at next photon on det1
                    else:
                        q = 1 # dt larger than maximum correlation width. stop. 
            else:
                q = 1 # end of photon stream on det1 reached. stop.

    # correlate det1 with det0 (positive time differences)
    lim1=0
    for i0 in range(len(times1)):
        t0 = times1[i0]
        i1 = 0
        q = 0
        while q == 0:
            if lim1 + i1 < len(times0):
                dt = times0[lim1+i1]-t0
                if dt < 0:
                    lim1 = lim1 + 1
                else:
                    binnr = int(dt/g2res)
                    if binnr < nrbins:
                        g2[nrbins - 1 - binnr] += 1
                        #if microtimes0[lim1+i1] > blindB and microtimes1[i0] > blindB:
                        #    g2B[nrbins - 1 - binnr] += 1
                        #if microtimes0[lim1+i1] > blindC and microtimes1[i0] > blindC:
                        #    g2C[nrbins - 1 - binnr] += 1
                        i1 = i1 + 1
                    else:
                        q = 1
            else:
                q = 1
                                
    g2tlist = np.arange(g2res*dtmicro*nrbins,g2restime)*1e9
    plt.plot(g2tlist,g2)
    #plt.plot(g2tlist,g2B)
    #plt.plot(g2tlist,g2C)
    plt.title('g(2) correlation')
    plt.xlabel('delay (ns)')
    plt.ylabel('occurence (a.u.)')
    plt.ylim([0,max(g2)])
    plt.show()

    return(g2tlist,g2,g2restime,nrbins)

def Autocor1APD(times0,nrbins=200):
    i0=0
    i1=0
    lim1=0
    g2 = np.zeros(2*nrbins)
    #g2B = np.zeros(2*nrbins)
    #g2C = np.zeros(2*nrbins)
    #blindB = 2e-9
    #blindC = 5e-9

    g2res = 1
    #blindB = blindB/tmicro
    #blindC = blindC/tmicro

    # correlate det0 with det1 (positive time differences)
    for i0 in range(len(times0)):
        t0 = times0[i0]
        i1 = 0
        q = 0
        while q == 0: 
            if lim1 + i1 < len(times0): # check if we've already reached end of photon stream on det1
                dt = times0[lim1+i1]-t0 
                if dt < 0: # find index lim1 of first photon on det1 that came after photon i0 on det0
                    lim1 = lim1 + 1
                else:
                    binnr = int(dt/g2res) # calculate binnr that corresponds to dt
                    if binnr < nrbins: # check if time difference is already large enough to stop correlation
                        g2[nrbins + binnr] += 1 # increase counter in corresponding bin by one
                        #if microtimes0[i0] > blindB and microtimes1[lim1+i1] > blindB:poi
                            #g2B[nrbins + binnr] += 1
                        #if microtimes0[i0] > blindC and microtimes1[lim1+i1] > blindC:
                            #g2C[nrbins + binnr] += 1
                        i1 = i1 + 1 # look at next photon on det1
                    else:
                        q = 1 # dt larger than maximum correlation width. stop. 
            else:
                q = 1 # end of photon stream on det1 reached. stop.

    
                                
    g2tlist = np.arange(-g2res*dtmicro*(nrbins-0.5),g2res*dtmicro*nrbins,nrbins*g2res)*1e9
    plt.plot(g2tlist,g2)
    #plt.plot(g2tlist,g2B)
    #plt.plot(g2tlist,g2C)
    plt.title('g(2) correlation')
    plt.xlabel('delay (ns)')
    plt.ylabel('occurence (a.u.)')
    plt.ylim([0,max(g2)])
    plt.show()

    return(g2tlist,g2,g2restime,nrbins)
def eVtonm(x):
    return 1240/x
def nmtoeV(x):
    return 1240/x

def Findtshift(Freq,Vpp,Voffset,calibcoeffs,macrocyclelist,dtmacro,matchrange=(500,570),shiftrange=(-6e-4,-2e-4),steps=30,Debugmode=False):
    InVoltagenew_c=nb.jit(nopython=True)(InVoltagenew) #compile to C to speed it up
    threshlow=1/Freq/4
    threshhigh=3/Freq/4
    #Sort microtimes in two halves
    Z = np.logical_and(threshlow<(macrocyclelist*dtmacro),(macrocyclelist*dtmacro)<= threshhigh)
    tforward=macrocyclelist[np.where(Z)]
    tbackward=macrocyclelist[np.where(np.logical_not(Z))]
    histbinnumber = 100 # 608 was for the entire range. For a matchrange of 520 to 590, this should be 4 times as small than the original to prevent aliasing
    #First coarse sweep
    # matchrange=(500, 570) #Wavelengthrange in which it should match. Maybe exclude the boundaries a bit
    tshift=np.zeros(steps)
    autocorr=np.zeros(steps)
    for k in tqdm(range(0,steps)):
        tshift[k]=shiftrange[0]+(shiftrange[1]-shiftrange[0])*k/steps
        lamforward=calibcoeffs[1]+InVoltagenew_c(tforward*dtmacro,Freq,Vpp,Voffset,tshift[k])*calibcoeffs[0]
        lambackward=calibcoeffs[1]+InVoltagenew_c(tbackward*dtmacro,Freq,Vpp,Voffset,tshift[k])*calibcoeffs[0]
        [ylistforward,xlistforward] = np.histogram(lamforward,histbinnumber,range=matchrange)
        # tlistforward = (xlistforward[:-1]+0.5*(xlistforward[1]-xlistforward[0]))
        [ylistbackward,xlistbackward] = np.histogram(lambackward,histbinnumber,range=matchrange)
        # tlistbackward = (xlistbackward[:-1]+0.5*(xlistbackward[1]-xlistbackward[0]))
        autocorr[k]=np.sum(ylistforward*ylistbackward)
    if Debugmode==True:
        plt.figure()
        plt.plot(tshift,autocorr,'.')
    optimumshift=tshift[np.argmax(autocorr)]
    if Debugmode==True:
        tshifttest=optimumshift
        lamforward=calibcoeffs[1]+InVoltagenew(tforward*dtmacro,Freq,Vpp,Voffset,tshifttest)*calibcoeffs[0]
        lambackward=calibcoeffs[1]+InVoltagenew(tbackward*dtmacro,Freq,Vpp,Voffset,tshifttest)*calibcoeffs[0]
        [ylistforward,xlistforward] = np.histogram(lamforward,50,range=matchrange)
        tlistforward = (xlistforward[:-1]+0.5*(xlistforward[1]-xlistforward[0]))
        [ylistbackward,xlistbackward] = np.histogram(lambackward,50,range=matchrange)
        tlistbackward = (xlistbackward[:-1]+0.5*(xlistbackward[1]-xlistbackward[0]))
        plt.figure()
        plt.plot(tlistforward,ylistforward)
        plt.plot(tlistbackward,ylistbackward)
    return optimumshift

def Easyhist(rawdata,lowestbin,highestbin,stepsize):
    edges=np.linspace(lowestbin-stepsize/2,highestbin+stepsize/2,int(round((highestbin-lowestbin+stepsize)/stepsize))+1)
    wavelbins=np.linspace(lowestbin,highestbin,int(round((highestbin-lowestbin)/stepsize))+1)
    histdata=np.histogram(rawdata,bins=edges)
    return wavelbins,histdata[0],edges


def fitspectra(binnedspectra,wavelengths,startfit,endfit,Debugmode=False):

    timeaverage = np.sum(binnedspectra,axis=1)
    lormod = LorentzianModel(prefix='Lor_')

    pars = lormod.guess(timeaverage-np.min(timeaverage), x=wavelengths)
    
    constmod = ConstantModel(prefix='Const_') 
    pars.update(constmod.make_params())
    
    mod = lormod + constmod
    
    init = mod.eval(pars, x=wavelengths)
    out = mod.fit(timeaverage-np.min(timeaverage), pars, x=wavelengths)
    
    
    plt.figure()
    plt.plot(wavelengths,timeaverage-np.min(timeaverage),label='best fit')
    # plt.plot(wavelengths, out.init_fit, 'k--', label='initial fit')
    plt.plot(wavelengths,out.best_fit,label='experimental data')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.legend(loc=0)
    
    
    print('time averaged fit results',out.fit_report()) #with the outputs obtained here, fill in the next section as guess
    fitreport=out.fit_report()
    
    amplitude = float(fitreport.split('amplitude:')[-1].split('+')[0])
    # amplitudeerr = float(fitreport.split('amplitude:')[-1].split('+/-')[1].split('(')[0])
    center = float(fitreport.split('center:')[-1].split('+')[0])
    # centererr = float(fitreport.split('center:')[-1].split('+/-')[1].split('(')[0])
    sigma = float(fitreport.split('sigma:')[-1].split('+')[0])
    # sigmaerr = float(fitreport.split('sigma:')[-1].split('+/-')[1].split('(')[0])
    

    
    pars_amplitude = amplitude
    pars_center = center#these values are generated based on the above section. Look at print fitreport
    pars_sigma = sigma
    
    
    peakamplitude = np.zeros(endfit-startfit)
    peakamplitudeerr = np.zeros(endfit-startfit)
    peakcenter = np.zeros(endfit-startfit)
    peakcentererr = np.zeros(endfit-startfit)
    peaksigma = np.zeros(endfit-startfit)
    peaksigmaerr = np.zeros(endfit-startfit)
    
    if Debugmode==True:    
        plt.figure()
    
    for spec in tqdm(range(startfit,endfit)):
        lormod = LorentzianModel()
        pars = lormod.guess(Yfilteredrebin[:,spec]-np.min(Yfilteredrebin[:,spec]), x=wavelengths1)
        pars['amplitude'].set(value=pars_amplitude)
        pars['center'].set(value=pars_center)
        pars['sigma'].set(value=pars_sigma)
        
        # pars['fwhm'].set(value=14, min=5, max=20)
        # pars['height'].set(value=100, min=0, max=400)
        
        constmod = ConstantModel(prefix='Const_')
        pars.update(constmod.make_params())
        mod = lormod + constmod
        
        init = mod.eval(pars, x=wavelengths1)
        out = mod.fit(Yfilteredrebin[:,spec]-np.min(Yfilteredrebin[:,spec]), pars, x=wavelengths1)
        
        fitreportspec = out.fit_report()
        
        peakamplitude[spec] = float(fitreportspec.split('amplitude:')[-1].split('+')[0])
        peakamplitudeerr[spec] = float(fitreportspec.split('amplitude:')[-1].split('+/-')[1].split('(')[0])
        peakcenter[spec] = float(fitreportspec.split('center:')[-1].split('+')[0])
        peakcentererr[spec] = float(fitreportspec.split('center:')[-1].split('+/-')[1].split('(')[0])
        peaksigma[spec] = float(fitreportspec.split('sigma:')[-1].split('+')[0])
        peaksigmaerr[spec] = float(fitreportspec.split('sigma:')[-1].split('+/-')[1].split('(')[0])
        
        
        if Debugmode==True:
            
            plt.plot(wavelengths1,Yfilteredrebin[:,spec]-np.min(Yfilteredrebin[:,spec]))
            # plt.plot(wavelengths1, init, 'k--', label='initial fit')
            plt.plot(wavelengths1,out.best_fit,label='best fit spec '+str(spec))
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Intensity')
            # plt.legend()
            plt.show()
            
    return peakamplitude,peakamplitudeerr,peakcenter,peakcentererr,peaksigma,peaksigmaerr




def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
    """Add a vertical color bar to an image plot, which is aligned to the figure. Simply use add_colorbar(im) with im representing imshow"""
    divider = axes_grid1.make_axes_locatable(im.axes)
    width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect)
    pad = axes_grid1.axes_size.Fraction(pad_fraction, width)
    current_ax = plt.gca()
    cax = divider.append_axes("right", size=width, pad=pad)
    plt.sca(current_ax)
    return im.axes.figure.colorbar(im, cax=cax, **kwargs)

    
def gaussian(x,A,mean,sigma,background):
    return A**2 / (sigma*np.sqrt(2*np.pi))*np.exp(-1 / 2 * (np.square((x-mean)/sigma**2)))+background


def rebin(arr, binning,):
    # reshapes the array. If you increase the binning, then your array will be smaller because binning means how many things you would like to average over
    # calculates the mean of a certain array to produce a binning by reshaping the array
    numberspectra,numberwavelengths = arr.shape[1],arr.shape[0]
    shape = arr.reshape(int(numberspectra/binning),numberwavelengths,binning)
    mean =  np.mean(shape,axis=2)
    meanreshaped = mean.reshape(numberwavelengths,int(numberspectra/binning))

    return meanreshaped


     

def spectralcorrelation(spectra1,spectra2,wavelengthsspectra1,wavelengthsspectra2,taus,plot):
    """
    

    Parameters
    ----------
    spectra1 : TYPE Array
        Data to be correlated with spectra2. Make sure to use a shape of (amount of wavelengths, amount of time bins)
    spectra2 : TYPE Array
        Data to be correlated with spectra1. Make sure to use a shape of (amount of wavelengths, amount of time bins)
    wavelengths : TYPE Array
        All the wavelengths of interest. Is used to display the photon pair correlation map.
    taus : TYPE list of taus that you are interested to see 
        DESCRIPTION.
    plot : TYPE String of text.
        Type of plot you are interested to see. You can choose between 'corr' (correlation map), 'cov' (covariance map), 'norm' (normalization map), 'all' (all maps)

    Returns
    -------
    covariance : TYPE Array
        Covariance of the calculates taus. 
    normalization : TYPE Array
        Normalization based on the standard deviations of the two types of lists. Calculates the outer product of the two times the length of the list (in time direction)
    correlation : TYPE Array
        Correlation (covariance divided by the normalization)

    Notes
    -------
    By choosing a part of spectra1 and spectra2, you can get postselection (and the average and normalization is adjusted on that)
    """




    covariance = np.zeros((len(taus),len(spectra1),len(spectra1))) #saves the covariances, correlations for different taus
    normalization = np.zeros((len(taus),len(spectra1),len(spectra1)))
    correlationtemp = np.zeros((len(taus),len(spectra1),len(spectra1)))
    normlambda1 = np.zeros((len(taus),len(spectra1)))
    normlambda2 = np.zeros((len(taus),len(spectra1)))
            
    for i in tqdm(range(len(taus))):
        tau = taus[i]
        for t in range(0,len(spectra1[0])-tau):
            covariance[i] += np.outer(spectra1[:,t],spectra2[:,t+tau])
            normlambda1[i,:] += spectra1[:,t]
            normlambda2[i,:] += spectra2[:,t+tau]
        normalization[i] = np.outer(normlambda1[i],normlambda2[i])*len(spectra1[0])+1 #the idea of the +1 was simply having to many dark counts which are averaged out and resulted in almost division by 0 errors. Nevertheless this also does not seem to work really properly actually.
        correlationtemp[i] = np.divide(covariance[i],normalization[i])
    
    correlation = np.zeros((len(taus),len(spectra1),len(spectra1))) 
    #here I use the correction for the decaying component in the spectra. Basically, due to non-overlapping stuff this creates a worse correlation than actually true. In order to circumvent that, either the data along the wavelength axes should be increased, or you have to do some correction (which are the lines below)
    for i in tqdm(range(len(taus))):
        tau = taus[i]
        for t in range(0,len(spectra1[0])-tau):
            correlation[i] = correlationtemp[i]/(1-tau/len(spectra1[0]))
            
    minwavspec1 = np.min(np.delete(wavelengthsspectra1.ravel(),np.where(wavelengthsspectra1.ravel()<=1))) #done to select the wavelength on the correct axis
    maxwavspec1 = np.max(np.delete(wavelengthsspectra1.ravel(),np.where(wavelengthsspectra1.ravel()<=1))) # I dont want to construct a map with all zeros
    minwavspec2 = np.min(np.delete(wavelengthsspectra2.ravel(),np.where(wavelengthsspectra2.ravel()<=1)))
    maxwavspec2 = np.max(np.delete(wavelengthsspectra2.ravel(),np.where(wavelengthsspectra2.ravel()<=1)))
    
    if plot=='corr': #with the string you can select which graphs you would like to see
        for i in range(len(taus)):
            tau = taus[i]
            
            plt.figure()
            plt.imshow(correlation[i],extent=[minwavspec1,maxwavspec1,maxwavspec2,minwavspec2])
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Wavelength (nm)')
            plt.title('Correlation map. tau = '+str(taus[i]))
            # if savefig=True: #attempt to save figures on the fly
                # plt.savefig('E:/Martijn/ETH/results/20200310_PM111_specdiffusion/QD2/Correlation_map_tau'+str(tau)+'_excitation',dpi=800) 
    elif plot=='cov':
        for i in range(len(taus)):
            tau = taus[i]
            
            plt.figure()
            plt.imshow(covariance[i],extent=[minwavspec1,maxwavspec1,maxwavspec2,minwavspec2])
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Wavelength (nm)')
            plt.title('Covariance map. tau = '+str(tau))
    elif plot=='norm':
        for i in range(len(taus)):
            tau = taus[i]
            
            plt.figure()
            plt.imshow(normalization[i],extent=[minwavspec1,maxwavspec1,maxwavspec2,minwavspec2])
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Wavelength (nm)')
            plt.title('Normalization map. tau = '+str(tau)) 
    
    elif plot=='all':
        for i in range(len(taus)):
            tau = taus[i]
            
            plt.figure()
            plt.imshow(correlation[i],extent=[minwavspec1,maxwavspec1,maxwavspec2,minwavspec2])
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Wavelength (nm)')
            plt.title('Correlation map. tau = '+str(taus[i]))
        
        for i in range(len(taus)):
            tau = taus[i]
            
            plt.figure()
            plt.imshow(covariance[i],extent=[minwavspec1,maxwavspec1,maxwavspec2,minwavspec2])
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Wavelength (nm)')
            plt.title('Covariance map. tau = '+str(tau))
        
        for i in range(len(taus)):
            tau = taus[i]
            
            plt.figure()
            plt.imshow(normalization[i],extent=[minwavspec1,maxwavspec1,maxwavspec2,minwavspec2])
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Wavelength (nm)')
            plt.title('Normalization map. tau = '+str(tau))
            
    else: #means that you dont want to see any plot
        pass
        
    return covariance, normalization, correlation  
  

def pearsoncorrelation(spectra1,spectra2,wavelengthsspectra1,wavelengthsspectra2,taus,plot):
    """
    

    Parameters
    ----------
    spectra1 : TYPE Array
        Data to be correlated with spectra2. Make sure to use a shape of (amount of wavelengths, amount of time bins)
    spectra2 : TYPE Array
        Data to be correlated with spectra1. Make sure to use a shape of (amount of wavelengths, amount of time bins)
    wavelengths : TYPE Array
        All the wavelengths of interest. Is used to display the photon pair correlation map.
    taus : TYPE list of taus that you are interested to see 
        DESCRIPTION.
    plot : TYPE String of text.
        Type of plot you are interested to see. You can choose between 'corr' (correlation map), 'cov' (covariance map), 'norm' (normalization map), 'all' (all maps)

    Returns
    -------
    covariance : TYPE Array
        Covariance of the calculates taus. 
    normalization : TYPE Array
        Normalization based on the standard deviations of the two types of lists. Calculates the outer product of the two times the length of the list (in time direction)
    correlation : TYPE Array
        Correlation (covariance divided by the normalization)

    Notes
    -------
    By choosing a part of spectra1 and spectra2, you can get postselection (and the average and normalization is adjusted on that)
    """
    #at the moment I still need to think about whether to let the function crash when the time dimension is not the same length for spectra1 and spectra2
    spectra1mean = np.mean(spectra1,axis=1) #subtract mean of the entire series to better observe fluctuations around the mean
    spectra2mean = np.mean(spectra2,axis=1)
    
    spectra1corr= np.zeros(spectra1.shape)
    spectra2corr= np.zeros(spectra2.shape)
    
    stdev1 = np.zeros((len(taus),len(spectra1)))
    stdev2 = np.zeros((len(taus),len(spectra2)))
    
    for j in range(0,len(spectra1)):
        spectra1corr[j,:] = spectra1[j,:]-spectra1mean[j] #mean is subtracted for each wavelength specific. Note that when plotting mean subtracted spectra you can not consider the gaussian curve anymore.
        
    for j in range(0,len(spectra2)):        
        spectra2corr[j,:] = spectra2[j,:]-spectra2mean[j]
        
    for i in tqdm(range(len(taus))):
        tau = taus[i]
        for j in range(0,len(spectra1)):
        # for t in range(0,len(spectra1corr[0])-tau):    
            stdev1[i,j] = np.std(spectra1[j,0:len(spectra1corr[0])-tau]) #standard deviation is calculated for the intensities across the time direction
   
    for i in tqdm(range(len(taus))):
        tau = taus[i]
        for j in range(0,len(spectra2)):
            stdev2[i,j] = np.std(spectra2[j,tau:len(spectra2corr[0])])
    

    covariance = np.zeros((len(taus),len(spectra1),len(spectra2))) #saves the covariances, correlations for different taus
    normalization = np.zeros((len(taus),len(spectra1),len(spectra2)))
    correlationtemp = np.zeros((len(taus),len(spectra1),len(spectra2)))
            
    for i in tqdm(range(len(taus))):
        tau = taus[i]
        for t in range(0,len(spectra1corr[0])-tau):
            covariance[i] += np.outer(spectra1corr[:,t],spectra2corr[:,t+tau])
      
        normalization[i] = np.outer(stdev1[i],stdev2[i])*len(spectra1[0])
        correlationtemp[i] = np.divide(covariance[i],normalization[i])
    
    correlation = np.zeros((len(taus),len(spectra1),len(spectra2))) 
    #here I use the correction for the decaying component in the spectra. Basically, due to non-overlapping stuff this creates a worse correlation than actually true. In order to circumvent that, either the data along the wavelength axes should be increased, or you have to do some correction (which are the lines below)
    for i in tqdm(range(len(taus))):
        tau = taus[i]
        for t in range(0,len(spectra1corr[0])-tau):
            correlation[i] = correlationtemp[i]/(1-tau/len(spectra1[0]))
            
    minwavspec1 = np.min(np.delete(wavelengthsspectra1.ravel(),np.where(wavelengthsspectra1.ravel()<=1))) #done to select the wavelength on the correct axis
    maxwavspec1 = np.max(np.delete(wavelengthsspectra1.ravel(),np.where(wavelengthsspectra1.ravel()<=1))) # I dont want to construct a map with all zeros
    minwavspec2 = np.min(np.delete(wavelengthsspectra2.ravel(),np.where(wavelengthsspectra2.ravel()<=1)))
    maxwavspec2 = np.max(np.delete(wavelengthsspectra2.ravel(),np.where(wavelengthsspectra2.ravel()<=1)))
    
    if plot=='corr': #with the string you can select which graphs you would like to see
        for i in range(len(taus)):
            tau=taus[i]
            if tau==0:
                vmax0corr=np.nanmax(np.delete(correlation.ravel(),np.where(correlation.ravel()>=0.98))) #done to remove the whole =1 diagonal visualization and allows for better visualization of the plot
                # print(vmax0corr)
                plt.figure()
                plt.imshow(correlation[i],extent=[minwavspec1,maxwavspec1,maxwavspec2,minwavspec2],vmax=vmax0corr)
                plt.colorbar()
                plt.gca().invert_yaxis()
                plt.xlabel('Wavelength (nm)')
                plt.ylabel('Wavelength (nm)')
                plt.title('Correlation map. tau = '+str(taus[i]))
                # if savefig=True: #attempt to save figures on the fly
                # plt.savefig('E:/Martijn/ETH/results/20200310_PM111_specdiffusion/QD2/Correlation_map_tau'+str(tau)+'_excitation',dpi=800) 
                
            else:
                plt.figure()
                plt.imshow(correlation[i],extent=[minwavspec1,maxwavspec1,maxwavspec2,minwavspec2])
                plt.colorbar()
                plt.gca().invert_yaxis()
                plt.xlabel('Wavelength (nm)')
                plt.ylabel('Wavelength (nm)')
                plt.title('Correlation map. tau = '+str(taus[i]))
            # if savefig=True: #attempt to save figures on the fly
                # plt.savefig('E:/Martijn/ETH/results/20200310_PM111_specdiffusion/QD2/Correlation_map_tau'+str(tau)+'_excitation',dpi=800) 
    elif plot=='cov':
        for i in range(len(taus)):
            tau = taus[i]
            
            plt.figure()
            plt.imshow(covariance[i],extent=[minwavspec1,maxwavspec1,maxwavspec2,minwavspec2])
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Wavelength (nm)')
            plt.title('Covariance map. tau = '+str(tau))
    elif plot=='norm':
        for i in range(len(taus)):
            tau = taus[i]
            
            plt.figure()
            plt.imshow(normalization[i],extent=[minwavspec1,maxwavspec1,maxwavspec2,minwavspec2])
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Wavelength (nm)')
            plt.title('Normalization map. tau = '+str(tau)) 
    
    elif plot=='all':
        for i in range(len(taus)):
            tau = taus[i]
            
            plt.figure()
            plt.imshow(correlation[i],extent=[minwavspec1,maxwavspec1,maxwavspec2,minwavspec2])
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Wavelength (nm)')
            plt.title('Correlation map. tau = '+str(taus[i]))
        
        for i in range(len(taus)):
            tau = taus[i]
            
            plt.figure()
            plt.imshow(covariance[i],extent=[minwavspec1,maxwavspec1,maxwavspec2,minwavspec2])
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Wavelength (nm)')
            plt.title('Covariance map. tau = '+str(tau))
        
        for i in range(len(taus)):
            tau = taus[i]
            
            plt.figure()
            plt.imshow(normalization[i],extent=[minwavspec1,maxwavspec1,maxwavspec2,minwavspec2])
            plt.colorbar()
            plt.gca().invert_yaxis()
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('Wavelength (nm)')
            plt.title('Normalization map. tau = '+str(tau))
            
    else: #means that you dont want to see any plot
        pass
        
    return covariance, normalization, correlation

    
def wavelengthcorrelation(spectra1,spectra2,exptime):
    """
    

    Parameters
    ----------
    spectra1 : Array of spectra to be correlated with spectra2. make sure that the shape of the array has wavelengths,time
        Array.
    spectra2 : Array of spectra to be correlated with spectra1. make sure that the shape of the array has wavelengths,time
        Array.

    Returns
    -------
    norm : array with intensities, correlated for the whole time range
    delaytime : A variable that shifts the correlation graph to almost match completely with tau=0. This can be done because the np.correlate function is to be interpeted as how similar the spectra look after a certain time

    """
    
    spectra1 = spectra1.ravel()
    spectra2 = spectra2.ravel()
    spectra1norm = (spectra1 - np.mean(spectra1)) / (np.std(spectra1) * (np.sqrt(len(spectra1)-1)))
    spectra2norm = (spectra2 - np.mean(spectra2)) / (np.std(spectra2) * (np.sqrt(len(spectra2)-1)))
    covariance = np.correlate(spectra1-np.mean(spectra1),spectra2-np.mean(spectra2),mode='same')
    correlation = np.correlate(spectra1norm, spectra2norm,mode='same')
    
    shiftt = np.linspace(-exptime/2,exptime/2,len(spectra1))
    correlationcorr = np.zeros(len(correlation))
    
    for j in range(len(correlationcorr)):
        correlationcorr[j] = correlation[j]/(1-np.abs(shiftt[j]/exptime))
        
    return correlation, shiftt, covariance, correlationcorr

def intensitycorrelation(spectra1,spectra2,exptime):
    """
    

    Parameters
    ----------
    spectra1 : Array of spectra to be correlated with spectra2. make sure that the shape of the array has wavelengths,time
        Array.
    spectra2 : Array of spectra to be correlated with spectra1. make sure that the shape of the array has wavelengths,time
        Array.

    Returns
    -------
    norm : array with intensities, correlated for the whole time range
    delaytime : A variable that shifts the correlation graph to almost match completely with tau=0. This can be done because the np.correlate function is to be interpeted as how similar the spectra look after a certain time
    spectra1sum, summed intensities without the correlation value
    spectra2sum, summed intensities without the correlation value
    """
    
    # spectra1 = spectra1.ravel()
    # spectra2 = spectra2.ravel()
    spectra1sum = np.sum(spectra1,axis=0)
    spectra2sum = np.sum(spectra2,axis=0)
    spectra1norm = (spectra1sum - np.mean(spectra1sum)) / (np.std(spectra1sum) * (np.sqrt(len(spectra1sum)-1)))
    spectra2norm = (spectra2sum - np.mean(spectra2sum)) / (np.std(spectra2sum) * (np.sqrt(len(spectra2sum)-1)))
    covariance = np.correlate(spectra1sum - np.mean(spectra1sum),spectra2sum - np.mean(spectra2sum),mode='same')
    correlation = np.correlate(spectra1norm, spectra2norm,mode='same')
    shiftt = np.linspace(-exptime/2,exptime/2,len(spectra1[0]))
    correlationcorr = np.zeros(len(correlation))
    for j in range(len(correlationcorr)):
        correlationcorr[j] = correlation[j]/(1-np.abs(shiftt[j]/exptime))
    return correlation, shiftt, spectra1sum, spectra2sum, covariance, correlationcorr


def wavelengthaverage(wavelengths,array):
    """Computes the average and find its location in the wavelength list"""
    #is a bit slow
    sum1=np.zeros(1)
    sum2=np.zeros(1)
    wavaverage=np.zeros(1)
    for j in range(len(array[0])): #this calculates the average wavelength depending using some kind of wavelength average
        for i in range(len(wavelengths)):
            temp = wavelengths[i]*array[i,j]
            sum1+=temp
            temp2 = array[i,j]
            sum2+=temp2
    wavaverage = sum1/sum2

    idx = (np.abs(wavelengths - wavaverage)).argmin()
    return wavaverage,idx,wavelengths[idx]

def fouriertransform(polardata,thetas):
    if polardata.ndim==2: #this 
                #this function works when the total amount of sampling points on the inner radius is the same as on the outer radius. This does not work when its not because the array then either gets distorted or you have a lot of zeros which may mess up the FT.
        
        fouriertaudata = [] # in order to retain the imaginary part the data must be calculted in a list. The data is then transformed to an array
        fouriertautheta = np.zeros(polardata.shape)
    
        
        for i in range(len(polardata)): #thus does it for all taus
            # datapoints = len(polardata[0][i])
            datapoints = len(thetas)
            fouriertransformdata = []
            # for j in range(len(polardata[i])):
            # fourtemp = fftpack.fft(polardata[i,:])
            fourtemp = fftpack.fft(polardata[i,:])/datapoints #is a normalization factor
            fouriertransformdata.append(fourtemp)
                # fouriertransformdataarr = np.asarray(fouriertransformdata)
            fouriertautheta[i,:] = fftpack.fftfreq(datapoints) * (datapoints)
    
            fouriertaudata.append(fouriertransformdata)
            
        fouriertaudata = np.asarray(fouriertaudata)
        
        fouriertauthetafftshift = np.zeros(polardata.shape)
        fouriertaudatarealfftshift = np.zeros(polardata.shape)
        fouriertaudataimagfftshift = np.zeros(polardata.shape)
        
        for i in range(len(polardata)):
            # for j in range(len(polardata[i])):
            fouriertauthetafftshift[i] = np.fft.fftshift(fouriertautheta[i])
            fouriertaudatarealfftshift[i] = np.fft.fftshift(fouriertaudata[i].real)
            fouriertaudataimagfftshift[i] = np.fft.fftshift(fouriertaudata[i].imag)
            
        
    elif polardata.ndim==3:
        #this function works when the total amount of sampling points on the inner radius is the same as on the outer radius. This does not work when its not because the array then either gets distorted or you have a lot of zeros which may mess up the FT.
        
        fouriertaudata = [] # in order to retain the imaginary part the data must be calculted in a list. The data is then transformed to an array
        fouriertautheta = np.zeros(polardata.shape)
    
        
        for i in range(len(polardata)): #thus does it for all taus
            # datapoints = len(polardata[0][i])
            datapoints = len(thetas)
            fouriertransformdata = []
            for j in range(len(polardata[i])):
                # fourtemp = fftpack.fft(polardata[i,j,:])
                fourtemp = fftpack.fft(polardata[i,j,:])/datapoints #is a normalization factor
                fouriertransformdata.append(fourtemp)
                # fouriertransformdataarr = np.asarray(fouriertransformdata)
                fouriertautheta[i,j,:] = fftpack.fftfreq(datapoints) * (datapoints)
    
            fouriertaudata.append(fouriertransformdata)
            
        fouriertaudata = np.asarray(fouriertaudata)
        
        fouriertauthetafftshift = np.zeros(polardata.shape)
        fouriertaudatarealfftshift = np.zeros(polardata.shape)
        fouriertaudataimagfftshift = np.zeros(polardata.shape)
        
        for i in range(len(polardata)):
            for j in range(len(polardata[i])):
                fouriertauthetafftshift[i,j] = np.fft.fftshift(fouriertautheta[i,j])
                fouriertaudatarealfftshift[i,j] = np.fft.fftshift(fouriertaudata[i,j].real)
                fouriertaudataimagfftshift[i,j] = np.fft.fftshift(fouriertaudata[i,j].imag)
                
    return fouriertautheta, fouriertaudata,fouriertauthetafftshift, fouriertaudatarealfftshift,fouriertaudataimagfftshift



def cartesiantopolarcorrelation(cartesiandata,origin,datapoints):
    """transforms cartesian data into polar data on a rectangular grid."""
    #converts cartesian covariance or correlation maps to 'rolled out' radii vs theta maps
    if datapoints%2==0:
        inputdata = np.zeros((len(cartesiandata),datapoints,datapoints))
        thetas = np.linspace(-np.pi,np.pi,datapoints,endpoint=False)
        radius = np.zeros((datapoints,datapoints))
        theta = np.zeros((datapoints,datapoints))


        for i in range(datapoints): #Radius and theta are only calculated once since the spectra all have the same radii and theta, because I chose the center point to be the same for all
            for j in range(datapoints):
                radius[i,j]=((i-int(datapoints/2))**2+(j-int(datapoints/2))**2)**(1/2)
                theta[i,j] = np.arctan2(i-int(datapoints/2),j-int(datapoints/2)) #calculates the radii and thetas for the data set when centered around the value you are interested in
         
        maximalradius=int(np.max(radius)) 
        indexx = np.zeros((len(cartesiandata),int(maximalradius),datapoints))
        indexy = np.zeros((len(cartesiandata),int(maximalradius),datapoints))
        radii = np.linspace(0,int(maximalradius),int(maximalradius))   
        polardata = np.zeros((len(inputdata),int(maximalradius),datapoints))
        
        for t in range(len(inputdata)):
            inputdata[t] = cartesiandata[t][origin-int(datapoints/2):origin+int(datapoints/2),origin-int(datapoints/2):origin+int(datapoints/2)]
            for i in range(0,int(maximalradius)): #if you are going to want sampling that has an increased value in the radius driction you should implement a for loop here
                
                indexx[t,i,:] = radii[i]*np.cos(thetas)
                indexy[t,i,:] = radii[i]*np.sin(thetas)
            
            # for i in range(0,int(maximalradius[t])):
                for k in range(datapoints):
                    polardata[t,i,k] = cartesiandata[t][int(round(indexx[t,i,k]))+origin][int(round(indexy[t,i,k]))+origin] # basically it searches for when the index matches pretty similarly the index around the origin.
                    # temp1 = polardata[t,i,k] = cartesiandata[t][int(indexx[t,i,k])+origin][int(indexy[t,i,k])+origin] 
                    # temp2 = polardata[t,i,k] = cartesiandata[t][int(indexx[t,i,k])+origin+1][int(indexy[t,i,k])+origin] 
                    # temp3 = polardata[t,i,k] = cartesiandata[t][int(indexx[t,i,k])+origin][int(indexy[t,i,k])+origin+1]  
                    # temp4 = polardata[t,i,k] = cartesiandata[t][int(indexx[t,i,k])+origin+1][int(indexy[t,i,k])+origin+1] 
                    # polardata[t,i,k] = np.mean([temp1,temp2,temp3,temp4]) # so with these you can basically have an average of the closes values in the data. Some figures did not show significant changes
    else:
        raise ValueError('datapoints must be a multiple of two so that it is centered around even values')
        
    
 
    return inputdata,polardata,thetas,maximalradius
    #so it returns inputdata that is cropped symmetricaly around the chosen origin with the number of datapoints being an input parameter

def wienerprocess(centerstart,scaling,length):
    x = centerstart+scaling*np.cumsum(np.random.randn(length)) #wiener process is defined in the sense that brownian motion makes independent gaussian steps at each point. The cumulative sum of independent normal random varaibles represents brownian motion.
    # y = centerstart+scaling*np.cumsum(np.random.randn(length))
    return x

def find_origin(normalizationdata,guessorigin,wavelengths,prominence=10,width=15):
    #might have to play a little with prominence and width due to trion peak
    temp = np.sum(normalizationdata,axis=0)
    peaks,rest= scipy.signal.find_peaks(temp,width=width,prominence=prominence)
    minimalvalue = np.argmin(np.abs(wavelengths[peaks]-guessorigin))
    peakprominent = peaks[minimalvalue]
    return peakprominent

def Fouriercomponentvstau(fourierdata, fourierangles, selectedradii):
    #less of a mess in script when components are matched
    component1theta = np.zeros((len(fourierdata),len(selectedradii)))
    component1data = np.zeros((len(fourierdata),len(selectedradii)))
    component2theta = np.zeros((len(fourierdata),len(selectedradii)))
    component2data = np.zeros((len(fourierdata),len(selectedradii)))
    component3theta = np.zeros((len(fourierdata),len(selectedradii)))
    component3data = np.zeros((len(fourierdata),len(selectedradii)))    
    for i in range(len(fourierdata)):
        for k in range(0,len(selectedradii)):
            selectedradius=selectedradii[k]
            for j in range(len(fourierdata[0][i])):
                if fourierangles[i][k][j]==0:
                
                    component1theta[i][k]= fourierangles[i][selectedradius][j]
                    component1data[i][k] = fourierdata[i][selectedradius][j]
                    
                elif fourierangles[i][k][j]==1:
                
                    component2theta[i][k] = fourierangles[i][selectedradius][j]
                    component2data[i][k] = fourierdata[i][selectedradius][j]
                    
                elif fourierangles[i][k][j]==2: # I think this one is the most interesting for the imaginairy part.
                
                    component3theta[i][k] = fourierangles[i][selectedradius][j]
                    component3data[i][k] = fourierdata[i][selectedradius][j]
    return component1theta,component1data,component2theta,component2data,component3theta,component3data
    
def find_originsyntheticdata(normalizationdata,guessorigin,wavelengths,prominence=10,width=15):
    #due to non amplitude fluctuations which are a little bit random and therefore hard to simulate it actually is the other way around
    temp = np.sum(-normalizationdata,axis=0)
    peaks,rest= scipy.signal.find_peaks(temp,width=width,prominence=prominence)
    minimalvalue = np.argmin(np.abs(wavelengths[peaks]-guessorigin))
    peakprominent = peaks[minimalvalue]
    return peakprominent,peaks

def ljungbox(h,correlations,lags):
    #a measure of how intense the autocorrelation values are compared to the standarad deviations
    length = len(correlations)
    for i in range(h):
        temp = correlations[i]**2/(length-lags[i])
    sum1 = np.sum(temp)  

def pearsoncorrelation1D(mean):
    #mean subtraction of fitted peak maxima. Generates normalized correlations and the 95% confidence interval
    mean =mean.ravel()
    meansub =  mean-np.mean(mean)
    corr = np.correlate(meansub,meansub,mode='same')
    stdev = np.std(mean)
    normcorr = corr/stdev**2/len(mean)
    return normcorr, corr, stdev*2/len(mean)**(1/2)


# Numba approach
def repeatvector(vecin,repeattimes):
    return np.repeat(vecin,repeattimes) 

#% Parallel Loops
def repeatvecparallel(k):
    return(np.matlib.repmat(calibcoeffs[1]+InVoltage(data[19][range(data[22][k+tau],data[22][k+tau+1]-1)]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0],len(data[22])-tau,1))
