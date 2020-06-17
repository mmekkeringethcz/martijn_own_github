# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:54:34 2020

@author: Mekkering
"""

#
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
from scipy import fftpack
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
from lmfit.models import LorentzianModel, GaussianModel, VoigtModel, LinearModel, ConstantModel
#mpl.rcParams['figure.dpi']= 300
import socket #enables to find computer name to make less of a mess with folders
# %matplotlib auto

def fitspectra(binnedspectra,wavelengths,startfit,endfit,model,Debugmode=False):

    timeaverage = np.sum(binnedspectra,axis=1)
    if model=='Gauss':
        lormod = GaussianModel(prefix='Gauss_')
    elif model=='Lor':
        lormod = LorentzianModel(prefix='Lor_')
    elif model=='Voigt':
        lormod = VoigtModel(prefix='Voigt_')
    

    pars = lormod.guess(timeaverage, x=wavelengths)
    
    constmod = ConstantModel(prefix='Const_') 
    pars.update(constmod.make_params())
    
    mod = lormod + constmod
    
    init = mod.eval(pars, x=wavelengths)
    out = mod.fit(timeaverage, pars, x=wavelengths)
    
    
    plt.figure()
    plt.plot(wavelengths,timeaverage,label='experimental data')
    # plt.plot(wavelengths, out.init_fit, 'k--', label='initial fit')
    plt.plot(wavelengths,out.best_fit,label='best fit')
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
        pars = lormod.guess(binnedspectra[:,spec]-np.min(binnedspectra[:,spec]), x=wavelengths)
        pars['amplitude'].set(value=pars_amplitude)
        pars['center'].set(value=pars_center)
        pars['sigma'].set(value=pars_sigma)
        
        # pars['fwhm'].set(value=14, min=5, max=20)
        # pars['height'].set(value=100, min=0, max=400)
        
        constmod = ConstantModel(prefix='Const_')
        pars.update(constmod.make_params())
        mod = lormod + constmod
        
        init = mod.eval(pars, x=wavelengths)
        out = mod.fit(binnedspectra[:,spec]-np.min(binnedspectra[:,spec]), pars, x=wavelengths)
        
        fitreportspec = out.fit_report()
        
        peakamplitude[spec] = float(fitreportspec.split('amplitude:')[-1].split('+')[0])
        peakamplitudeerr[spec] = float(fitreportspec.split('amplitude:')[-1].split('+/-')[1].split('(')[0])
        peakcenter[spec] = float(fitreportspec.split('center:')[-1].split('+')[0])
        peakcentererr[spec] = float(fitreportspec.split('center:')[-1].split('+/-')[1].split('(')[0])
        peaksigma[spec] = float(fitreportspec.split('sigma:')[-1].split('+')[0])
        peaksigmaerr[spec] = float(fitreportspec.split('sigma:')[-1].split('+/-')[1].split('(')[0])
        
        
        if Debugmode==True:
            
            plt.plot(wavelengths,binnedspectra[:,spec]-np.min(binnedspectra[:,spec]))
            # plt.plot(wavelengths1, init, 'k--', label='initial fit')
            plt.plot(wavelengths,out.best_fit,label='best fit spec '+str(spec))
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
            # if tau==0:
            vmax0corr=np.nanmax(np.delete(correlation.ravel(),np.where(correlation.ravel()>=0.95))) #done to remove the whole =1 diagonal visualization and allows for better visualization of the plot
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
                
            # else:
            #     plt.figure()
            #     plt.imshow(correlation[i],extent=[minwavspec1,maxwavspec1,maxwavspec2,minwavspec2])
            #     plt.colorbar()
            #     plt.gca().invert_yaxis()
            #     plt.xlabel('Wavelength (nm)')
            #     plt.ylabel('Wavelength (nm)')
            #     plt.title('Correlation map. tau = '+str(taus[i]))
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
    return peakprominent,peaks

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
    return sum1

def pearsoncorrelation1D(mean):
    #mean subtraction of fitted peak maxima. Generates normalized correlations and the 95% confidence interval
    mean =mean.ravel()
    meansub =  mean-np.mean(mean)
    corr = np.correlate(meansub,meansub,mode='same')
    stdev = np.std(mean)
    normcorr = corr/stdev**2/len(mean)
    delaytau=np.linspace(-len(mean)/2,len(mean)/2,len(mean))
    return normcorr, corr, stdev*2/len(mean)**(1/2),delaytau-0.5


# Numba approach
def repeatvector(vecin,repeattimes):
    return np.repeat(vecin,repeattimes) 

#% Parallel Loops
def repeatvecparallel(k):
    return(np.matlib.repmat(calibcoeffs[1]+InVoltage(data[19][range(data[22][k+tau],data[22][k+tau+1]-1)]*dtmacro,Freq,Vpp,Voffset,Verror)*calibcoeffs[0],len(data[22])-tau,1))



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

def Findtshift(Freq,Vpp,Voffset,calibcoeffs,macrocyclelist,dtmacro,matchrange=(500,570),shiftrange=(-6e-4,-2e-4), histbinnumber = 100,steps=30,Debugmode=False):
    InVoltagenew_c=nb.jit(nopython=True)(InVoltagenew) #compile to C to speed it up
    threshlow=1/Freq/4
    threshhigh=3/Freq/4
    #Sort microtimes in two halves
    Z = np.logical_and(threshlow<(macrocyclelist*dtmacro),(macrocyclelist*dtmacro)<= threshhigh)
    tforward=macrocyclelist[np.where(Z)]
    tbackward=macrocyclelist[np.where(np.logical_not(Z))]
    # histbinnumber = 100 # 608 was for the entire range. For a matchrange of 520 to 590, this should be 4 times as small than the original to prevent aliasing
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
        [ylistforward,xlistforward] = np.histogram(lamforward,histbinnumber,range=matchrange)
        tlistforward = (xlistforward[:-1]+0.5*(xlistforward[1]-xlistforward[0]))
        [ylistbackward,xlistbackward] = np.histogram(lambackward,histbinnumber,range=matchrange)
        tlistbackward = (xlistbackward[:-1]+0.5*(xlistbackward[1]-xlistbackward[0]))
        plt.figure()
        plt.plot(tlistforward,ylistforward)
        plt.plot(tlistbackward,ylistbackward)
    return optimumshift
