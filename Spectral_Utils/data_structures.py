"""
This library is intended to contain useful data structures for quick plotting and visualization + easy standard processing of time series converted in ATS format
"""


import numpy as np
import matplotlib.pyplot as plt
import pywt
import scipy.signal as scisi
import ATS_Utils.ATSIO as atsio
import Spectral_Utils.QCutils as qc
import os
import Spectral_Utils.TimeFreqTools as tft

class Timeseries(object):
    
    def __init__(self, filname):
        
        self.header= atsio.ReadATSHead(filname)
        self.data= atsio.ReadATSData(filname)* self.header['LSBVal'][0]
        self.startTime= self.header['StartSecond'][0]
        self.nsamp= self.header['NSamp'][0]
        self.fc= self.header['SRate'][0]
        self.dt= 1./ self.fc
        self.timeline= np.arange(self.startTime,self.startTime+(self.nsamp*self.dt),self.dt)
        self.sitename= os.path.basename(filname).split('_')[0]
        self.cal= None
        
    #Power/amplitude spectral density
    def PSD(self, nsamp, M, over, ptype='a'):
        
        if ptype=='a':
            print "\nCalculating amplitude spectral density.."
            self.fk, self.ASD= qc.AutoSpectrum(self.data[:nsamp], M, over, np.hamming(M), self.fc, type= ptype, calibration= self.cal)
        else:
            print "\nCalculating power spectral density.."
            self.fk, self.PSD= qc.AutoSpectrum(self.data[:nsamp], M, over, np.hamming(M), self.fc, type= 'p', calibration= self.cal)
         
    #Get calibration   
    def get_cal(self, M, cal_path):
        
        self.cal= qc.calc_frsp(M, self.fc, cal_path)
    
    #Coherence/transfer function 
    def CohTF(self, M, over, filen, cal2=None):
        
        #Get data for the second channel
        sig2= Timeseries(filen)
        if cal2!=None:
            sig2.get_cal(M, cal2)
        
        #Calculate Coh2 and TF
        self.fk, self.Coh2, self.TF= qc.CoherenceFast(self.data, self.data, self.fc, M, over, np.hamming(M), calibration1=self.cal, calibration2=sig2.cal )
    
    #Discrete Wavelet transform
    def DWT(self, nsamp, wtype='db', mode='per', level=None):
        
        if level!=None:
            self.dwt_coef= pywt.wavedec(self.data[:nsamp], pywt.wavelist(family='db')[1], mode, level)
        else:
            self.dwt_coef= pywt.wavedec(self.data[:nsamp], pywt.wavelist(family='db')[1], mode)
    
    #Multi-notch filter
    def notch(self, numsamp, freqs):
        
        self.data[:numsamp]= qc.notch(self.data[:numsamp], freqs, self.fc)
    
    #Remove DC
    def remDC(self):
        
        self.mean= np.mean(self.data)
        self.data -= self.mean
    
    #Detrending
    def detrend(self):
        
        self.data = scisi.detrend(self.data)
    
    #Wigner-Ville time-frequency distribution
    def wdf(self, numsamp, kdt=1):
        
        self.wdf= tft.WDF(self.data[:numsamp], self.fc)
        
        self.freq_wdf= np.fft.fftfreq(2*numsamp,self.dt)[:numsamp+1]
        self.freq_wdf[-1] *= -1
        self.time_wdf= np.arange(0,numsamp*self.dt,self.dt*kdt)
    
    #Discrete wavelet transform coefficients - plotting
    def plot_dwt(self, plot=False):
        
        fig=plt.figure(figsize=(12,12))
        ax= fig.add_subplot(111)
        ax.grid(True),
        
        for i in self.dwt_coef:
            ax.plot(i,'-'),
        
        if plot:
            plt.show()
        else:
            return fig
    
    #Wigner-Ville time-frequency distribution - plotting
    def plot_wdf(self, plot=False):
        from matplotlib.colors import LogNorm
        
        fig= plt.figure(figsize=(12,12))
        ax1= fig.add_subplot(111)
        
        maps= ax1.pcolormesh( self.freq_wdf, self.time_wdf, np.abs(self.wdf) ,cmap='gray', norm=LogNorm()),
        ax1.set_title("Wigner-Ville distribution")
        ax1.set_xlabel("Frequency [Hz]"),
        ax1.set_ylabel("Time [s]"),
        ax1.axis('tight'),
        plt.colorbar(maps[0],ax=ax1)
        
        if plot:
            plt.show()
        else:
            return fig
    
    #Coherence/transfer function plotting
    def plot_cohTF(self, MT_type='L', plot=False):
        
        #Frequency + NaN masking
        mask= self.fk<= 0.4*(self.fc/2.)
        #Remove first 4 FFT bin
        mask[:4]= False
        
        #NAN masking
        mask= np.logical_and(mask, self.Coh2!=np.nan)
        
        fig= plt.figure( figsize=(8.2677,11.6929) )
        fig.suptitle("Site: "+str(self.sitename)+", Channels: " , size='large')
        ax= fig.add_subplot(311)
        ax2= fig.add_subplot(312)
        ax3= fig.add_subplot(313)
        plt.subplots_adjust(bottom = 0.1, left  = 0.125 )
        ax.set_xlabel("Periods [s]"),
        ax.set_ylabel("Squared Coherence"),
        ax.grid(True)
        ax2.set_xlabel("Periods [s]"),
        ax2.set_ylabel("Amplitude TF"),
        ax2.grid(True)
        ax3.set_xlabel("Periods [s]"),
        ax3.set_ylabel("Phase TF [deg]"),
        ax3.grid(True)
        
        ax.semilogx( 1./self.fk[mask], self.Coh2[mask], '-' ),
        if MT_type=='L':
            ax.set_xlim(1e-5,1e2),
            ax2.set_xlim(1e-5,1e2),ax2.set_ylim( 10**(-0.5),10**0.5 ),
            ax3.set_xlim(1e-5,1e2),ax3.set_ylim( -25,25 )
        else:
            ax2.set_ylim( 10**(-3),10**1 )
            
        ax.set_ylim( 0,1.1 ),
        ax2.loglog( 1./self.fk[mask], np.abs(self.TF[mask]), '-' ),
        ax3.semilogx( 1./self.fk[mask], np.rad2deg( np.angle(self.TF[mask]) ), '-' ),
        
        if plot:
            plt.show()
        else:
            return fig
            
        
    #Time series/spectra plotting
    def plot(self, numsamp=None, stype='time', ax= None, show= False):
        
        if ax== None:
            fig= plt.figure(figsize=(12,12))
            ax= fig.add_subplot(111)
            
            if stype=='time':
            
                if numsamp!=None:
                    ax.plot(self.timeline[:numsamp], self.data[:numsamp], 'k-'), ax.grid(True),
                else:
                    ax.plot(self.timeline, self.data, 'k-'), ax.grid(True),
            
            else:
                ax.set_title('Sitename: '+self.sitename)
                ax.loglog(self.fk, self.PSD,'r-'), ax.grid(True),
                ax.set_xlabel('Frequency [Hz]'),ax.set_ylabel('Power spectral density [mV^2/Hz]'),
            
        else:
            if stype=='time':
            
                if numsamp!=None:
                    ax.plot(self.timeline[:numsamp], self.data[:numsamp], 'b-'), ax.grid(True),
                else:
                    ax.plot(self.timeline, self.data, 'b-'), ax.grid(True),
            
            else:
                ax.set_title('Sitename: '+self.sitename)
                ax.loglog(self.fk, self.PSD,'k-'), ax.grid(True),
                ax.set_xlabel('Frequency [Hz]'),ax.set_ylabel('Power spectral density [mV^2/Hz]'),
            
        if show:
            plt.show()
        else:
            return ax