# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:24:03 2020

@author: Mekkering
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

          
         