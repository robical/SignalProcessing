import numpy as np
import matplotlib.pyplot as plt
import Spectral_Utils.QCutils as qc
import copy

def hstrk(data, **kwargs):
    """
    Histogram with sample tracking
    --> useful to derive sample statistics and do robust processing of data
    """
    optparams={
               'numbins': 10,
               'stack': False,  
               }
    
    optparams.update(kwargs)
    
    #Get data limits
    mind= np.min(data)
    maxd= np.max(data)
    
    #Creating the base set of bins for the histogram
    #this base will span the entire domain covered by the signal
    num= optparams['numbins']
    delta= (maxd - mind)/num
    bins= np.zeros( (num,2) )
    bins[0,0]= mind
    bins[0,1]= bins[0,0]+delta
    bins[-1,0]= maxd- delta
    bins[-1,1]= maxd
    i=1
    while(i<(num-1)):
        bins[i,0]= bins[i-1,1]
        bins[i,1]= bins[i,0] + delta
        i+=1
        
    #calculate histogram with sample tracking
    histo= np.zeros( np.size(bins,0), dtype=np.double )
    histo_ind={}
    #for each sample in the slice
    for ind,i in enumerate(data):
        #Loop on each bin --> dicotomic search here
        kinit= 0
        klast= num
        
        while(kinit!=(klast-1) ):
            kk= np.int32( (kinit+klast)/2 )
            if i< bins[kk,0]:
                klast= kk
            elif i>bins[kk,0]:
                kinit= kk
            else:
                break
                  
        #if the value is contained in the kk-th bin, than put it in the dictionary at the entry KK
        if( i<=bins[kinit,1] and i>=bins[kinit,0] ):
            histo[kinit] += 1 #value counter for kk-th bin
            #if the bin already exist, just add the tracking infos
            if(histo_ind.has_key(kinit)):
                histo_ind[kinit].append(ind) #value index tracker for values contained in kk-th bin
            else:
                #If the bin doesn't exist, create an entry and add the tracking info
                histo_ind[kinit]=[]
                histo_ind[kinit].append(ind)
                    
    return histo_ind, histo, bins



def despiker(data, **kwargs):
    """
    General simple despiking algorithm in time
    
    """
    
    optparams={
               'showdif': False,
               'M': 100,
               'wind': 'ones',
               }
    
    #Get first and second central difference to check for spikes
    data_d= np.gradient(data)
    data_dd= np.gradient(data_d)
    
    if optparams['showdif']:
        fig=plt.figure(figsize=(12,12))
        ax=fig.add_subplot(1,1,1)
        ax.plot(np.abs(data),'b'),
        ax.plot(np.abs(data_d+ np.mean(data) ),'r'),
        ax.plot(np.abs(data_dd+ np.mean(data) ),'g'), ax.grid(True)
        ax.legend(['Original','First diff +mean','Second diff+mean'])
        plt.show()
    
    #Slicing the first derivative of the signal
    M= optparams['M']
    window= eval( 'np.%s(%i)' % (optparams['wind'], M) ) #evaluate and create the externally selected window at runtime
    [sig,L]=qc.windowing(data_d, M, 0.5, window)
    
    #Signal slicing
    [sig_da,La]= qc.windowing(data, M, 0.5, window)
    
    #Calculations of running statistics: STD and MEDIAN for each slice
    run_std= np.zeros(L)
    run_mu= np.zeros(L)
    for i in range(int(L)):
        run_std[i]= np.std(sig[:,i])
        run_mu[i]= np.median(sig[:,i])
    
    #Calculate global STD of the first derivative
    glob_std= np.std(data_d)
        
    #Test of using standard deviation as threshold
    if optparams['showdif']:
        fig=plt.figure(figsize=(12,12))
        ax=fig.add_subplot(1,1,1)
        ax.plot(run_std,'b-*'),
        ax.plot(glob_std*np.ones(L),'r-')
        ax.grid(True),
        ax.legend(['local STD - fd','global STD - as thresh'])
        plt.show()
    
    #The running standard deviation can be used to track the spikes
    bad_list=[]
    for ind,i in enumerate(run_std):
        if(i>glob_std):
            bad_list.append(ind)
    
    #Select the bad segments on the first difference of the signal slice
    for k in bad_list:
        #Calculate histogram with sample tracking for each bad slice
        histo_ind, histo= hstrk(sig[:,k])
        
        #Select the bin that contains the lowest number of entry
        histo[histo==0]=1e3 #to not count zero counts bin
        bad_bin= np.argmin(histo)
        
        #All the samples corresponding to this bin become interpolated values
        for ll in histo_ind[bad_bin]:
            if ll<(M-1) and ll>0:
                sig_da[ll,k]= (sig_da[ll+1,k]-sig_da[ll-1,k])/2.
            elif ll==0:
                sig_da[ll,k]= sig_da[ll+1,k]/2.
            else:
                sig_da[ll,k]= sig_da[ll-1,k]/2.
        
    data_new= qc.reconstruct(sig_da,0.5, window )
        
    return data_new

def median_despiker(data, L):
    """
    Simple running median despiker with support L
    """
    N= len(data)
    lowlim= (L-1)/2
    uplim= ( N-lowlim )
    data_c= data.copy()
    ind=1
    while(ind<N):
        if ind<lowlim:
            M= len(data[:ind])
            datas= np.sort(data[:ind])
            if np.mod(M,2)==0 and M>1:
                data_c[ind]= datas[M/2]
            elif np.mod(M,2)!=0 and M>1:
                data_c[ind]= datas[(M-1)/2 +1]
            else:
                data_c[ind]= datas[0]
                
        elif ind>uplim:
            M= len(data[ind:])
            datas= np.sort(data[ind:])
            if np.mod(M,2)==0 and M>1:
                data_c[ind]= datas[M/2]
            elif np.mod(M,2)!=0 and M>1:
                data_c[ind]= datas[(M-1)/2 +1]
            else:
                data_c[ind]= datas[0]
        else:
            datas= np.sort( data[ind-lowlim:ind+lowlim+1] )
            #substitutes with median
            data_c[ind]= datas[(L-1)/2 +1]
            
        ind+=1
        
    return data_c
        
            
        

if __name__=="__main__":
    
    import ATS_Utils.ATSIO as atsio
    import os
    import Spectral_Utils.QCutils as qc
    
    path_file= r"C:\Users\calandrinir\Documents\173_V01_C04_R000_THz_BL_64H.ats"
    
    head= atsio.ReadATSHead(path_file)
    data_in= atsio.ReadATSData(path_file)
    
    N= len(data_in)
    dt= (1./head['SRate'])
    t= np.arange(0,N*dt,dt)
    data_in= data_in*head['LSBVal']
    
    #Median despiking
    L=5
    data_out= median_despiker(data_in, L)
    
    #compare spectral content
    M= 1024
    over= 0.5
    wind= np.hamming(M)
    fc= head['SRate'][0]
    
    #spec_in= qc.spectrogram(data_in, M, over, wind, fc, plot=True, index_plot='T')
    
    fk, spO= qc.AutoSpectrum(data_in, M, over, wind, fc)
    fk, spD= qc.AutoSpectrum(data_out, M, over, wind, fc)
    
    fig= plt.figure( figsize=(12,12) )
    fig.suptitle('Trace: '+os.path.basename(path_file))
    ax= fig.add_subplot(211)
    ax.plot( t,data_in, 'r-', t,data_out, 'k-' ), ax.grid(True),
    ax.set_xlabel('Time [s]'), ax.set_ylabel('Amplitude [mV]'),
    ax.legend(['Original','Despiked'])
    
    ax2= fig.add_subplot(212)
    ax2.loglog( fk, spO, 'r-', fk, spD, 'k-' ), ax2.grid(True),
    ax2.set_xlabel('Frequency [Hz]'), ax2.set_ylabel('PSD [mV^2/Hz]'),
    ax2.legend(['Original','Despiked'])
    
    plt.show()
    
    