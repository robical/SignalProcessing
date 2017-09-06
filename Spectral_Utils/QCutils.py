# -*- coding: utf-8 -*-
"""
Created on Wed Feb 29 10:48:48 2012

Suite of tools for Quality Control of MT Time series and Spectral Analysis

@author: calandrinir
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, detrend

#Exception class hierachy

#Base class
class SpectraException(Exception):
    def __call__(self):
        pass
    
#SpectraException subclasses
class FilterNotImplemented(SpectraException):
    def __init__(self, message):
        self.msg= message
        
    def __call__(self):
        print self.msg
        
class WindowingError(SpectraException):
    def __init__(self, message):
        self.msg= message 
        
    def __call__(self):
        print self.msg
        
class SpectrumTypeException(SpectraException):
    def __init__(self, message):
        self.msg= message
        
    def __call__(self):
        print self.msg
        


#Base Class for MT related exception
class MTException(SpectraException):
    def __call__(self):
        pass

class ChannelError(MTException):
    def __init__(self, value):
        self.ChValue= value
        self.msg= "Channel error: "+str(self.ChValue)+",the channel you've selected, was not recognized"
        
    def __call__(self):
        print self.msg
        
class SensorError(MTException):
    def __init__(self, value1, value2):
        self.type= value1
        self.num= value2
        self.msg= "Sensor error: "+str(self.type)+" and "+str(self.num)+" are not in sensor list"
        
    def __call__(self):
        print self.msg

#It's not possible to have multiple constructors in Python so I've arranged to do it with the default values       
class ChopperError(MTException):
    def __init__(self, name=None, tipo=None):
        self.name= name
        self.ChName= tipo
        if(self.name!=None and self.ChName!=None):
            self.msg= "ChopperError: the chopper parameter is not set in time series "+self.name+" of type "+str(self.ChName)
        else:
            self.msg= "ChopperError: the chopper parameter is not set"
       
    def __call__(self):
        print self.msg 
        
class IndexTracker:
    """
    Class that defines a tracking object that can show slices of a Multidimensional object in an interactive graph,
    with the possibility to scroll the graph to look inside the ND-object slice by slice
    """
    def __init__(self, ax, X, x, index='T'):
        self.ax = ax
        self.ax.set_title('Use scroll wheel to navigate slices of the spectrogram')
        self.ax.grid(True)
        self.ax.set_xlabel('Frequency [Hz]'),
        self.ax.set_ylim(1e-6,1e-1)

        self.index= index
        self.X = X #y-axis
        self.x= x #x-axis
        
        if index=='T':
            _,self.sel = X.shape
            self.ind  = self.sel/2

            self.graph = self.ax.loglog(self.x, self.X[:,self.ind])
            self.update()
        else:
            self.sel,_ = X.shape
            self.ind  = self.sel/2

            self.graph = self.ax.semilogy(self.x, self.X[self.ind,:])
            self.update()
            

    def onscroll(self, event):
        #print ("%s %s" % (event.button, event.step))
        if event.button=='up':
            self.ind = np.clip(self.ind+1, 0, self.sel-1)
        else:
            self.ind = np.clip(self.ind-1, 0, self.sel-1)


        self.update()

    def update(self):
        if self.index=='T':
            self.graph[0].set_data( (self.x,self.X[:,self.ind]) )
            self.ax.set_ylabel('PSD [mV^2/Hz] for slice %s'%self.ind)
            self.graph[0].axes.figure.canvas.draw()
        else:
            self.graph[0].set_data( (self.x,self.X[self.ind,:]) )
            self.ax.set_ylabel('PSD [mV^2/Hz] in time, for frequency %s [Hz]'% self.x[self.ind])
            self.graph[0].axes.figure.canvas.draw()
        
        
def zeropad(data, Np):
    """
    Implements zero padding on both sides of the signal, for odd and even signal length
    """
    #Manual zero padding
    if(np.mod(Np,2)==0): #even
        data = np.fft.ifftshift(data) #used just for reordering
        data= np.concatenate( (np.zeros(Np/2), data, np.zeros(Np/2)) )
        data= np.fft.fftshift(data) #used just for reordering
    else: #odd
        data = np.fft.ifftshift(data) #used just for reordering
        data= np.concatenate( (np.zeros((Np-1)/2), data, np.zeros((Np-1)/2 +1)) )
        data= np.fft.fftshift(data) #used just for reordering
    
    return data


def antidiff(sig, step=1):
    """
    Implements the adjoint operator of the diff
    """
    
    out= np.zeros_like(sig)
    for ind,i in enumerate(sig[:-1:]):
        out[ind]= (i+sig[ind+1])*step
        
    out[ind+1]= out[ind]
    
    return out


def diff(sig, step=1.):
    """
    Implements a first difference operator that leaves the output
    the same length as the input --> time method, first difference
    """
    out= np.zeros_like(sig)
    for ind,i in enumerate(sig[:-1:]):
        out[ind]= (sig[ind+1]-i)/(step)
    
    out[ind+1]=out[ind]
    
    return out


def f_diff(sig,fc, plot=False, **kwargs):
    """
    Frequency domain discrete derivator
    
    NB: use non-causal derivator only for non real-time processing
    """
    
    optparams={
               'type': 'non-causal',
               }
    
    optparams.update(kwargs)
    
    N= sig.size #number of samples
    dt= 1./fc
    ftype= optparams['type']
        
    #construct derivator in frequency domain
    fk= np.arange(0,N/2.+1,1)
    finv= fk[::-1]
    fk= np.concatenate( (fk,-finv[1:]) )
        
    #spectrum calc
    S= np.fft.fft(sig, n=int(N) ) #zero padded FFT    
    
    #derivator construction
    if ftype=='non-causal':
        #non-causal derivator
        D= 1j*2*np.pi*fk
    elif ftype=='FIR_1':
        #first order derivator
        D= 1j*(2/dt)*np.sin(np.pi*fk)*np.exp(-1j*np.pi*fk)
    elif ftype=='IIR_CN':
        #Crank-Nicholson derivator: exact ideal phase, amplitude distorsion at higher frequencies wrt FIR (causal) first order filter
        D= 1j*(2/dt)*np.tan(np.pi*fk)
    else:
        raise Exception("Error: derivator filter type not recognized!")
        
    #convolution in freq domain and ifft
    sd= np.real( np.fft.ifft( S*D ) )
    
    if plot:
        
        if optparams['type']=='non-causal':
            m_amp= 'ko'
            m_phase= 'ro'
        else:
            m_amp= 'k-'
            m_phase= 'r-'
            
        
        fig= plt.figure(figsize=(10,10))
        fig.suptitle("Filter type: "+str(ftype)+" transfer function")
        ax= fig.add_subplot(211)
        ax.set_xlabel("Frequency [Hz]"),ax.set_ylabel("Amplitude")
        ax2= fig.add_subplot(212)
        ax2.set_xlabel("Frequency [Hz]"),ax2.set_ylabel("Phase [deg]")
        ax.plot( np.fft.fftshift(fk),np.fft.fftshift(np.abs(D)), m_amp, ms=10),ax.grid(True)
        ax2.plot( np.fft.fftshift(fk), np.fft.fftshift(np.rad2deg(np.angle(D))), m_phase, ms=10),ax2.grid(True)
        
        plt.show()
    
    
    return sd
    



def linreg(x,y):
    """
    Return coefficient of linear regression:
    1) Type 1: L2 ->  minimization of errors squared
    2) Type 2: L1 -> minimization of abs error (to do..)
    3) Type 3: Linf -> minimization of max error (to do..)
    """
    
    #L2 minimization
    
    #Create problem matrix
    N= np.size(x,0)
    T= np.zeros( (N,2) )
    
    T[:,0]= np.ones( N )
    T[:,1]= x
    TT= np.dot( T.T,T)
    
    g= np.dot( np.linalg.pinv(TT), np.dot(T.T,y) )
    
    m=g[1]
    c=g[0]
    
    return m,c


def delay(sig, fc, tau, over):
    """
    Delay a discrete signal sampled with frequency fc, with delay tau [s]
    
    INPUTS:
    -sig= signal
    -fc= sampling rate
    -tau= delay in [s]
    -over= oversampling factor
    
    OUTPUT:
    -result= delayed signal
    
    (TO BE COMPLETED)
    """
    
    #Done in time domain
    N= sig.size #effective duration of the signal
    NN= over*N #oversampling to smooth the delayer response
    dff= fc/float(NN) #frequency resolution on which the delayer transform will be calculated
    fk= np.arange( -NN/2,NN/2,1 )*dff #oversampled frequency axis
    
    #oversampled delayer transform
    Rk= np.exp(-1j*((2*np.pi*fk)/float(NN))*float(tau) )
    Rk= np.fft.fftshift(Rk)
    
    #antitransform to obtain the low-pass impulse response of the delayer in time
    filt= np.real(np.fft.ifft(Rk))

    return filt


def notch_lattice(sig, freqs, fc, **kwargs):
    """
    Notch filter stable implementation trough recursive lattice filtering
    """
    
    optparams={
               'notch_band': 3,
               }
    
    optparams.update(kwargs)
    
    N= sig.size
    
    #conversion to normalized pulsations
    ws= (freqs/fc)*2*np.pi
    
    #conversion of the bandwidth from [Hz] to normalized pulsation
    B= (optparams['notch_band']/fc)*2*np.pi
    
    #output vector initialization
    e1= np.zeros( N, dtype=np.double )
    
    for iif in ws:  
        #parameter calculation
        theta1= iif - (np.pi/2.)
        stheta2= (1.-np.tan(B/2.))/(1.+np.tan(B/2.))
        stheta1= np.sin(theta1)
        st= stheta1*(1.+stheta2)
        
        #Direct finite difference implementation of the filter
        i= N-3
        while(i>=0):
            e1[i]= (1./(2*stheta2))*( (1+stheta2)*(sig[i+2]+sig[i]) + (2*st)*(sig[i+1]-e1[i+1]) -2*e1[i+2] )
            i-= 1
        
            
    return e1
        
    
def pow_goertzel(sig,freqs,fc):
    """
    Estimate power spectrum level of selected frequencies
    using Goertzel algorithm
    """ 
    N= fc #number of FFT points in order to use K as a multiplier
    Nf= freqs.size
    Ns= sig.size
    power= np.zeros(Nf, dtype=np.double)
    
    start=0
    stop= N
    count=0
    while(stop<= (Ns-N) ):
        #Extract signal block
        sig_block= sig[:N]
        count += 1
        
        for indf,f in enumerate(freqs):
            if f<fc/2.:
                wk= 2*np.pi*(f/N)
                wr= np.cos(wk)
                c= 2*wr
                
                s1= 0.0
                s2= 0.0
                for k in range(N):
                    s= sig_block[k] + c*s1 - s2
                    s2= s1
                    s1= s
                    
                power[indf] += (s2**2+ s1**2 - c*s1*s2)
                
            else:
                print "One of the selected frequency is bigger than Nyquist!"
                continue
            
        start += N
        stop += N
    
    #Mean
    power= power/count
        
    return freqs,power


def notch(sig, freqs, fc, **kwargs):
    """
    Quick Notch filter: remove selected frequency from the signal
    """
    
    optparams={
               'Q': 0.99, #position of a complex conjugate pair of poles on the unit circle
               'balance': False, #doesn't notch frequencies, balance to neighborouring values of power spectra
               'plot': False, 
               }
    
    optparams.update(kwargs)
    
    import scipy.signal as sp
    
    Q= float(optparams['Q'])
    
    z = []
    p = []
    
    freq = np.array( sorted(freqs) )
    
    if optparams['balance']:
        #estimates of power line frequencies power
        freqs,pows= pow_goertzel(sig,freqs,fc)
        
        #estimates nearby frequencies power
        okfr= [45.]
        freqs, okpow= pow_goertzel(sig,okfr,fc)
        
        #TODO: complete power line compensation scheme using Goertzel algorithm for fast PSD computation for single frequencies
        
        
        
    
    omega_0 = 2*np.pi*freq/fc
    f1= np.exp(1j*omega_0)
    f1n= np.conj(f1)
    z.extend(f1)
    z.extend(f1n)
    p.extend(Q*f1)
    p.extend(Q*f1n)
    
    #Calculate polynomial coefficients from radixes
    b = np.atleast_1d(np.poly(z))
    a = np.atleast_1d(np.poly(p))
    
    #calculate initial condition
    sig_filtered = np.real( sp.lfilter(b, a, sig) )
    
    if optparams['plot']:
        #fig= plt.figure(figsize=(12,12))
        ax= plt.subplot2grid( (2,3), (0,0), rowspan=2, colspan=2 )
        ax.set_title("Z transform plot"),
        ax.set_xlabel("Re axis"),
        ax.set_ylabel("Im axis"),
        ax.grid(True),
        circ= plt.Circle( (0,0), radius= 1., color='g' )
        circ.set_facecolor("none")
        circ.set_edgecolor('black')
        ax.add_patch( circ ),
        
        #plot zeroes
        for i in z:
            ax.plot( np.real(i),np.imag(i), 'o', ms=10, mfc='blue',mec='blue' ),
            
        #plot poles
        for i in p:
            ax.plot( np.real(i),np.imag(i), 'x', ms=10, mfc='red', mec='red' ),
        
        #Fourier transform
        w= np.arange(0.0,np.pi,np.pi/500.)
        num= np.zeros_like(w, dtype=np.complex128)
        den= np.zeros_like(w, dtype=np.complex128)
        Nb= b.size
        Na= a.size
        
        #NUM
        for i in range(Nb):
            num += ( b[Nb-i-1]* np.exp(-1j*w*i) )
            
        #DEN
        for i in range(Na):
            den += ( a[Na-i-1]* np.exp(-1j*w*i) )
            
        ax2= plt.subplot2grid( (2,3), (0,2), rowspan=1, colspan=1 )
        ax2.plot( (w/(2*np.pi))*fc, 10*np.log10( np.abs(num/den) ), 'b-' ),ax2.grid(True),
        ax2.set_title("Amplitude [dB]"), ax2.set_xlabel("Freq. [Hz]"),
        ax3= plt.subplot2grid( (2,3), (1,2), rowspan=1, colspan=1 )
        ax3.plot( (w/(2*np.pi))*fc, np.angle(num/den), 'r-' ), ax3.grid(True),
        ax3.set_title("Phase [rad]"), ax3.set_xlabel("Freq. [Hz]"),
        
        plt.show()
        
    return sig_filtered
    

def windowing(signal,M,over,wind, **kwargs):
    """
    Precalculate the number of overlapped segments and return an
    np.array that has M rows and L columns, where the number of rows is the length of each
    subtrace extracted from the signal, and the number of columns is the number of subtraces extracted 
    """
    
    import scipy.signal as scisi
    import scipy
    import numpy.linalg as lnal
    
    # TODO: 1) Gaussianity test: t-TEST and/or Kolmogorov TEST 2) Test for stationarity of the time series
    # 
    # 
    #
    # Use the two operation above to make a raw selection of subsegments, to improve calculation
    # of transfer function and coherence
    
    #to mantain retro-compatibility
    #The detrend and removeDC option would be useful for spectral analysis
    optparams={
               'glob_rem_DC': True,
               'glob_detrend': False, #only if check is True
               'detrend_thresh': 5, #threshold in degrees used when checking for a linear trend
               'loc_rem_DC': True, #this should be used rarely
               'loc_detrend': True, #this should be used rarely
               'return_all': False, #activate to return also local_trend, local_DC and global_trend, global_DC if active
               'do_win': True, #do windowing
               'transform': False, #do RFFT 
               }
    
    optparams.update(kwargs)
    
    #Global mean removal + detrending
    if(optparams['glob_rem_DC']):
        glob_DC= np.mean(signal)
        signal= signal - glob_DC
        
    if(optparams['glob_detrend']):
        N= np.size(signal,0)
        
        #Linear fit of the entire trace
        (ar,_)= scipy.polyfit(np.arange(0,N,1),signal,1)
        
        #if the fit reveal a slope of more than +-5 degrees then detrend
        if( np.abs( np.rad2deg(np.arctan(ar)) )> optparams['detrend_thresh'] ):
            signal= scisi.detrend(signal)
    
    # transpose if the vector is row
    N=np.size(signal,0) #original signal length
    if(N==1):
        signal=signal.T
        N=np.size(signal,0) #original signal length
    
    #Calculate maximum number of section in the signal, with padding if necessary
    R= M*(1-over)
    L= 1+ np.floor( (N-M)/R )
    
    #Pre-allocate space for signal slices (mainly for time-frequency processing)
    #WARNING:  This part is VERY memory consuming if the routine that uses windowing is used with very large
    #          arrays.... consider using lowmem_win instead
    if not optparams['transform']:
        sig= np.zeros( (M,L), dtype=np.double )
    else:
        sig= np.zeros( (M/2.,L), dtype= np.complex128 )
    
    if optparams['return_all']:
        loc_DC= np.zeros( L )
        loc_trend= np.zeros( (L,2) )
        
    #Calculation of signal slices and windowing, eventually applying pre-processing to each subtrace
    start=0    
    for i in range(int(L)):
        stop=start+M
        temp= np.squeeze(signal[start:stop])       
        if(optparams['loc_rem_DC']):
            locDC= np.mean(temp)
            temp = temp- locDC
            if(optparams['return_all']):
                loc_DC[i]= locDC
            
        if(optparams['loc_detrend']):
            #Least square fit
            a= np.zeros( (M,2) ) #coeff matrix
            a[:,0]= np.ones(M)
            a[:,1]= np.arange(start,stop,1)
            xx,_,_,_= lnal.lstsq(a, temp) 
            temp= temp- (xx[0]*a[:,0] +xx[1]*a[:,1])
            
            if optparams['return_all']:
                loc_trend[i,0]=xx[0] #constant
                loc_trend[i,1]=xx[1] #slope
        
        if(optparams['do_win']): 
            temp=temp*wind
        
        if not optparams['transform']:
            sig[:,i]= temp
        else:
            sig[:,i]= np.fft.rfft(temp)[1:]
            
        start += R
    if optparams['return_all']:
        return [sig,L,loc_DC,loc_trend,glob_DC]
    else:    
        return [sig,L]
    
    
def lowmem_prepare(signal,M,over,wind, **kwargs):
    """
    Prepare the signal for a memory efficient windower for Autospectra, Cross-spectra,
    Coherence and Transfer function calculation with very BIG data
    
    INPUTS:
        -> signal: time series
        -> M: window length (maximum additional memory occupation induced by the processing)
        -> over: window overlap percentage ( e.g. 0.5 means 50% )
        -> wind: window (length M) that will be applied to each M-slice taken from the signal
        
    OUTPUTS:
        -> this function returns a generator, that each time grab a modified signal's window, ready to be processed
    """
    optparams={
               'glob_rem_DC': True,
               'glob_detrend': False, #only if check is True
               'detrend_thresh': 5, #threshold in degrees used when checking for a linear trend
               'loc_rem_DC': True, #this should be used rarely
               'loc_detrend': True, #this should be used rarely
               'return_all': False, #activate to return also local_trend, local_DC and global_trend, global_DC if active
               'do_win': True, #do windowing 
               }
    
    optparams.update(kwargs)
    
    N= np.size(signal,0)
    
    #Global mean removal + detrending
    if(optparams['glob_rem_DC']):
        glob_DC= np.mean(signal)
        signal -=  glob_DC
        
    if(optparams['glob_detrend']):
        import scipy.signal as scisi
        import scipy
        
        #Linear fit of the entire trace
        (ar,_)= scipy.polyfit(np.arange(0,N,1),signal,1)
        
        #if the fit reveal a slope of more than +-5 degrees then detrend
        if( np.abs( np.rad2deg(np.arctan(ar)) )> optparams['detrend_thresh'] ):
            signal= scisi.detrend(signal)
            
    #Calculate maximum number of section in the signal, with padding if necessary
    R= M*(1-over)
    L= 1+ np.floor( (N-M)/R )
    
    return L,R
    


def reconstruct(sig,over, wind,NN, **kwargs):
    """
    Reconstruct a signal decomposed in slices
    Note: if detrending and/or dc removal was applied to the original signal before or during windowing, it cannot be recovered during the 
    reconstruction, but it is possible for the user to supply the global and local informations as optional arguments to the function.
    """
    
    optparams= {
                'glob_DC': 0.0,
                'loc_DC': np.zeros( np.size(sig,1), dtype=np.double ),
                'glob_trend': None,
                'loc_trend': np.zeros( (np.size(sig,1),2) , dtype=np.double), 
                }
    
    optparams.update(kwargs)
    
    M= np.size(sig,0) #number of samples per slice
    L= np.size(sig,1) #number of slices
    Move= M*(1-over) #step in samples
    
    data= np.zeros( M*L*2, dtype=np.double) #total number of samples --> greater than input NN
        
    start=0
    for i in range(int(L)):
        stop= start+M
        
        #Eliminate windowing effect
        mask= [wind!=0.0]
        sig[:,i][mask]= sig[:,i][mask]/wind[mask]
        
        #Add local trend
        sig[:,i] += ( np.arange(start,stop,1)*optparams['loc_trend'][i,1] + np.ones(M)*optparams['loc_trend'][i,0] )
        
        #Add local DC
        data[start:stop] = ( sig[:,i] + optparams['loc_DC'][i] )
        start+= Move
    
    #trunk the data (inverse of zero padding)
    data= data[:NN]
    
    #Add global trend if present
    if optparams['glob_trend']!=None:
        data += ( np.arange(0,NN,1)*optparams['glob_trend'][0,1] + np.ones(NN)*optparams['glob_trend'][0,0] )
        
    #Add global DC if present
    data += optparams['glob_DC']
        
    return data


def cola_check(N,M,over,wind):
    """
    Routine that test for COLA condition
    the chosen window and hop size -> over
    """
    
    R= M*(1-over)
    L= 1+ np.floor( (N-M)/R ) 
    cola= np.zeros(N)
    istart=0
    i=0
    while(i<L):
        iend= istart+M
        cola[istart:iend] += wind
        istart= istart+ (M*(1-over))
        i +=1
        
    return cola


def STFT(signal,M,over,wind, fc):
    import scipy.signal as scisi
    
    #Pre-processing of signal
    N= signal.size
    signal -= np.mean(signal)
    signal= scisi.detrend(signal)
    
    #Pre-allocate space for the STFT
    dt= 1./fc
    R= M*(1-over)
    L= 1+ np.floor( (N-M)/R )
    
    STFT= np.zeros( (M,L), dtype=np.complex128 )
    fk= np.fft.fftshift( np.fft.fftfreq(M, dt) ) #discrete freq. axis
    Tw= np.arange(0, (L*M)*dt,M*dt) #discrete (slow) time axis
    start=0
    for i in range(int(L)):
        stop= start+M
        #preparing signal segment
        temp= signal[start:stop]
        temp -= np.mean(temp)
        temp = scisi.detrend(temp)
        #do FFT and save
        STFT[:,i]= np.fft.fftshift( np.fft.fft(temp) )
        start += R
        
    return STFT,fk,Tw    


def qqplot(data, **kwargs):
    """
    This routine will show a qq-plot of the data, assuming they are gaussian distributed.
    The method is general and can accomodate also non standardized data.
    """
    optparams={
               'plot': False,
               'dist': 'norm',
               'fit': True,
               }
    
    optparams.update(kwargs)
    
    from scipy.stats import probplot
    
    mean= np.mean(data)
    std= np.std(data)
    
    if optparams['plot']:
        res= probplot(data, sparams=(mean, std), dist=optparams['dist'], fit=optparams['fit'], plot=plt)
        plt.show()
        
        return res
    else:
        res= probplot(data, sparams=(mean, std), dist=optparams['dist'], fit=optparams['fit'], plot=None)
        
        return res
    

def spectraGen(sig,L):
    """
    Signal slices generator 
    --> the best would be, to avoid memory limits on the signal size, to create a generator that use PyTables to pop up pieces
    of the signal only when it's needed, having stored the entire signal on disk.
    """
    for i in range( int(L) ):
        yield sig[:,i]
        
def spectraGen_lowmem(sig,L,R,wind, loc_rem_DC= True, loc_detrend= True):
    """
    Low memory windowed slice generator
    """
    import scipy.signal as scisi
    
    M= wind.size
    start= 0
    for i in range(int(L)):
        stop = start+M
        
        temp= sig[start:stop].copy()
        
        if(loc_rem_DC):
            locDC= np.mean(temp)
            temp = temp- locDC
            
        if(loc_detrend):
            #Least square fit
            temp= scisi.detrend(temp)
        
        temp *= wind
        start += R
        
        yield temp
        
def phase_est(signal,M,over,wind,fc,**kwargs):
    """
    Estimate phase of a stochastic process
    
    It's based on a stationarity assumption
    """
    optparams={
               'glob_rem_DC': True,
               'glob_detrend': False,
               'loc_rem_DC': True,
               'loc_detrend': True,
               'unwrap': False,
               'calibration': None,
               'verbose': False,
               }
    
    optparams.update(kwargs)
    
    lowmem= False
    try:
        #Windowing
        [sig,L]= windowing(signal,M,over,wind, glob_rem_DC= optparams['glob_rem_DC'], glob_detrend=optparams['glob_detrend'],
                           loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
    except MemoryError:
        L,R= lowmem_prepare(signal,M,over,wind, glob_rem_DC= optparams['glob_rem_DC'], glob_detrend=optparams['glob_detrend'],
                           loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
        lowmem= True
        
    #Create positive frequency axis removing DC
    fk= np.fft.fftfreq( np.int(M), 1./fc)[:M/2+1]
    fk= fk[1:]
    fk[-1]= -fk[-1] #make Nyquist positive
    
    if optparams['verbose']:
        print "\nNumber of subsegments extracted: "+str(L)
        print "\nVariance reduction to expect: < "+str( np.sqrt(L) )
        
    if not lowmem:
        sgen= spectraGen(sig,L)
    else:
        #lowmem option
        sgen= spectraGen_lowmem(signal,L,R,wind, loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
        
    real_s= np.zeros( M/2 )
    imag_s= np.zeros( M/2 )
    if optparams['calibration']==None:
        #Phase calculation
        for i in sgen:
            real_s += np.real( np.fft.rfft( np.fft.fftshift(i) )[1:] )
            imag_s += np.imag( np.fft.rfft( np.fft.fftshift(i) )[1:] )
    else:
        #Calibrated Power spectral density calculation
        for i in sgen:
            real_s += ( ( np.real( np.fft.rfft(i)[1:] )*np.real(optparams['calibration']) )+\
            ( np.imag( np.fft.rfft(i)[1:] )*np.imag(optparams['calibration']) ) )/ \
            np.abs(optparams['calibration'])**2
            
            imag_s += ( np.imag(np.fft.rfft(i)[1:])*np.real(optparams['calibration']) )- \
            ( np.real( np.fft.rfft(i)[1:] )*np.imag(optparams['calibration']) )/ \
            np.abs(optparams['calibration'])**2
    
    real_s /= L
    imag_s /= L
    
    phase= np.arctan2( imag_s,real_s )
    
    return fk,phase
        
        

def AutoSpectrum(signal, M, over, wind, fc, **kwargs):
    """
    Calculates autopower spectrum for a signal using Welch modified periodogram method
    
    Default to power spectrum ['p' type]
    It's possible to select also 'a', for amplitude auto-spectral density
    
    The routine INPUTS are (ordered):
    - signal
    - M (window length)
    - over (window overlap)
    - wind (window used, must be of length M)
    - fc (Sampling rate)
    
    Optional args:
    - glob_rem_DC (boolean, remove DC from the whole signal)
    - glob_detrend (boolean,remove trends from the whole signal)
    - loc_rem_DC (boolean, remove DC from local slice)
    - loc_detrend (boolean, remove trends from local slice)
    - keepDC (boolean, return or not the DC component in the output)
    - type ('p' or 'a', type of spectra of interest: power or amplitude)
    - calibration (default to False, put in calibration function if you want the input to be calibrated)
    - verbose (boolean, show more diagnostic messages)
    
    OUTPUTS:
    - fk (frequency axis)
    - Spectrum (PSD or ASD)
    
    NB:  the routine automatically removes DC component
    """
    
    optparams={
               'glob_rem_DC': True,
               'glob_detrend': False,
               'loc_rem_DC': True,
               'loc_detrend': True,
               'type': 'p',
               'keepDC': False,
               'calibration': None,
               'verbose': False,
               }
    
    optparams.update(kwargs)
    
    
    lowmem= False
    try:
        #Windowing
        [sig,L]= windowing(signal,M,over,wind, glob_rem_DC= optparams['glob_rem_DC'], glob_detrend=optparams['glob_detrend'],
                           loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
    except MemoryError:
        L,R= lowmem_prepare(signal,M,over,wind, glob_rem_DC= optparams['glob_rem_DC'], glob_detrend=optparams['glob_detrend'],
                           loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
        lowmem= True
    
    #Create positive frequency axis removing DC
    fk= np.fft.fftfreq( np.int(M), 1./fc)[:M/2+1]
    if not optparams['keepDC']:
        fk= fk[1:]
        
    fk[-1]= -fk[-1] #make Nyquist positive
    
    if optparams['verbose']:
        print "\nNumber of subsegments extracted: "+str(L)
        print "\nVariance reduction to expect: < "+str( np.sqrt(L) )
    
    if optparams['type']=='p' or optparams['type']=='a':
        
        if not lowmem:
            sgen= spectraGen(sig,L)
        else:
            #lowmem option
            sgen= spectraGen_lowmem(signal,L,R,wind, loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
        
        if not optparams['keepDC']:
            Pspectrum= np.zeros( M/2 )
        else:
            Pspectrum= np.zeros( M/2+1 )
            
        win_en2= np.sum(wind**2)
        if optparams['calibration']==None:
            if not optparams['keepDC']:
                #Power spectral density calculation
                for i in sgen:
                    Pspectrum += ( np.abs( np.fft.rfft(i)[1:] )**2 / (fc*win_en2) )
            else:
                #Power spectral density calculation
                for i in sgen:
                    Pspectrum += ( np.abs( np.fft.rfft(i) )**2 / (fc*win_en2) )
        else:
            if not optparams['keepDC']:
                #Calibrated Power spectral density calculation
                for i in sgen:
                    Pspectrum += ( np.abs( np.fft.rfft(i)[1:] /(optparams['calibration']) )**2 / (fc*win_en2) )
            else:
                #Calibrated Power spectral density calculation
                for i in sgen:
                    Pspectrum += ( np.abs( np.fft.rfft(i) /(optparams['calibration']) )**2 / (fc*win_en2) )
            
        if np.mod(M,2)==0:
            if not optparams['keepDC']:
                #Since it's a semi-spectrum, all frequency components must be multiplied by 2, except for Nyquist, in case of M even
                Pspectrum[:-1]= Pspectrum[:-1]*2
            else:
                #Since it's a semi-spectrum, all frequency components must be multiplied by 2, except for Nyquist and zero, in case of M even
                Pspectrum[1:-1]= Pspectrum[1:-1]*2
        else:
            #In case M it's odd, all frequencies must be multiplied
            Pspectrum= Pspectrum*2
        
        #Mean Power spectral density
        Pspectrum = Pspectrum / L
        
        if optparams['type']=='p':
            #returns PSD
            return fk, Pspectrum
        else:
            #returns ASD
            return fk, np.sqrt(Pspectrum)
    
    else:
        raise SpectrumTypeException("Spectrum type: "+optparams['type']+" is not recognized!")
    
    
def CrossSpectrum(signal1, signal2, M, over, wind, fc, **kwargs):
    """
    Calculates CrossPower spectrum for a couple of signal using Welch modified periodogram method
    
    Default to power spectrum ['p' type]
    It's possible to select also 'a', for amplitude auto-spectral density
    
    The routine INPUTS are (ordered):
    - signal1
    - signal2
    - M (window length)
    - over (window overlap)
    - wind (window used, must be of length M)
    - fc (Sampling rate)
    
    Optional args:
    - glob_rem_DC (boolean, remove DC from the whole signal)
    - glob_detrend (boolean,remove trends from the whole signal)
    - loc_rem_DC (boolean, remove DC from local slice)
    - loc_detrend (boolean, remove trends from local slice)
    - type ('p' or 'a', type of spectra of interest: power or amplitude)
    - order ('same' or 'inverse', default to 'same', defines if you want the cross-spectrum to be calculated on the signal ordered as the input or switched)
    - calibration1 (default to False, put in calibration function if you want the input1 to be calibrated)
    - calibration2 (default to False, put in calibration function if you want the input2 to be calibrated)
    - verbose (boolean, show more diagnostic messages)
    
    OUTPUTS:
    - fk (frequency axis)
    - Amplitude of the Cross-spectral density
    - Phase of the Cross-spectral density (same as the cross-spectrum because the scale factor is purely real, so doesn't affect the phase)
    
    NB:  the routine automatically removes DC component
    NNB: you have to input calibration function with DC component removed!
    """
    
    optparams={
               'glob_rem_DC': True,
               'glob_detrend': False,
               'loc_rem_DC': True,
               'loc_detrend': False,
               'type': 'p',
               'order': 'same', #same as entered
               'calibration1': None,
               'calibration2': None,
               'verbose': False,
               }
    
    optparams.update(kwargs)
    
    lowmem= False
    try:
        [sig1,_]= windowing(signal1,M,over,wind, glob_rem_DC= optparams['glob_rem_DC'], glob_detrend=optparams['glob_detrend'],
                            loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
        [sig2,L]= windowing(signal2,M,over,wind, glob_rem_DC= optparams['glob_rem_DC'], glob_detrend=optparams['glob_detrend'],
                        loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
    except MemoryError:
        L,R= lowmem_prepare(signal1,M,over,wind, glob_rem_DC= optparams['glob_rem_DC'], glob_detrend=optparams['glob_detrend'],
                           loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
        lowmem= True
    
    if optparams['verbose']:
        print "\nNumber of subsegments extracted: "+str(L)
        print "\nVariance reduction to expect: < "+str( np.sqrt(L) )
    
    #Create positive frequency axis removing DC
    fk= np.fft.fftfreq(M, 1./fc)[:M/2+1]
    fk= fk[1:]
    fk[-1]= -fk[-1] #make Nyquist positive
    
    if optparams['type']=='p' or optparams['type']=='a':
        if not lowmem:
            sgen1= spectraGen(sig1,L)
            sgen2= spectraGen(sig2,L)
        else:
            sgen1= spectraGen_lowmem(signal1,L,R,wind, loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
            sgen2= spectraGen_lowmem(signal2,L,R,wind, loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
        
        CRspectrum= np.zeros( M/2 )+1j*np.zeros( M/2 ) #automatically removes DC
        win_en2= np.sum(wind**2)
        
        #Calculations
        if optparams['calibration1']==None and optparams['calibration2']==None:
            i=0
            while(i<L):
                CRspectrum += ( np.fft.rfft( next(sgen1) )[1:]* np.conj( np.fft.rfft( next(sgen2) )[1:] ) )/(win_en2*fc) # S1*S2^h
                i+=1
                
        elif optparams['calibration1']!=None and optparams['calibration2']==None:
            i=0
            while(i<L):
                CRspectrum += ( (np.fft.rfft( next(sgen1) )[1:]/optparams['calibration1'] )* np.conj( np.fft.rfft( next(sgen2) )[1:] ) )/(win_en2*fc) # S1*S2^h
                i+=1
                
        elif optparams['calibration1']==None and optparams['calibration2']!=None:
            i=0
            while(i<L):
                CRspectrum += ( np.fft.rfft( next(sgen1) )[1:] * np.conj( (np.fft.rfft( next(sgen2) )[1:]/optparams['calibration2']) ) )/(win_en2*fc) # S1*S2^h
                i+=1
        
        elif optparams['calibration1']!=None and optparams['calibration2']!=None:
            i=0
            while(i<L):
                CRspectrum += ( (np.fft.rfft( next(sgen1) )[1:]/optparams['calibration1'] )* np.conj( (np.fft.rfft( next(sgen2) )[1:]/optparams['calibration2']) ) )/(win_en2*fc) # S1*S2^h
                i+=1
                
        if optparams['order']=='same':
            return fk, np.abs( CRspectrum ), np.angle(CRspectrum)
        elif optparams['order']=='inverse':
            return fk, np.abs( CRspectrum ), np.angle( np.conj(CRspectrum) )
    else:
        raise SpectrumTypeException("Spectrum type: "+optparams['type']+" not recognized..")
        
def SetBandAveraging(fullfreqaxis, nf1, nfend, NFPD):
    
    minfreq = fullfreqaxis[nf1]
    maxfreq = fullfreqaxis[nfend]
    
    lminfreq = np.floor(np.log10(minfreq))
    lmaxfreq = np.ceil(np.log10(maxfreq))
    nfreq = (lmaxfreq - lminfreq)*NFPD + 1
    
    outfreqaxis = np.logspace(lminfreq, lmaxfreq, nfreq)
    mask = (outfreqaxis >= minfreq) & (outfreqaxis <= maxfreq)
    outfreqaxis = outfreqaxis[mask]
    nfreq = len(outfreqaxis)
    
    loutfreqaxis = np.log10(outfreqaxis)
    logedges = np.r_[loutfreqaxis[0],0.5*(loutfreqaxis[1:]+loutfreqaxis[:-1])]
    
    bins = np.searchsorted(logedges, np.log10(fullfreqaxis))-1
    
    baindices = []
    for ifreq in range(nfreq):
        ind = np.flatnonzero(bins == ifreq)
        ind = ind[(ind>=nf1)&(ind<=nfend)]
        baindices.append(ind)
    
    return outfreqaxis, baindices
    
    
def CoherenceFast(signal1, signal2, fc, M, over, wind, **kwargs):
    """
    Fast coherence and transfer function calculation
    
    Used by: 
        - pstselftest in ADU_Console
    
    For MT use:
        - the routine expects to receive each channel in [V] or [mV] for the electric field already normalized for dipole length
        - for magnetic channels, the routine expects to receive a channel in [V] --> then it corrects for the calibration function and convert it to [nT]
        - The SECOND input channel (signal2), is going to be conjugated during the calculation; this is relevant for tranfer function calculation
    
    The spectral quantities used are power spectral densities (so normalized for the window's energy) and Cross-spectral density
    
    Methods used:
        - Welch averaging for TF and Coherence calculation
        - Robust pre-selection: Daniell coherence averaging method in frequency domain
        - Band averaging: logarithmically equally spaced grid
    
    INPUTS:
    - signal1
    - signal2
    - fc: Sampling rate
    - M: window length
    - over: window overlap, expressed as fraction of unity (e.g. 0.5)
    - wind: window used, must be of length M
    
    
    Optional args:
    - glob_rem_DC: (True) Remove DC from the whole signal
    - glob_detrend: (False) Remove a linear trend from the whole signal
    - loc_rem_DC: (True) Remove DC from local slice
    - loc_detrend: (True) Remove trends from local slice
    - calibration1: (None) Put in calibration function if you want the input1 to be calibrated
    - calibration2: (None) Put in calibration function if you want the input2 to be calibrated
    - verbose: (False) Show more diagnostic messages
    - robust: (False) activate/deactivate coherence thresholding
    - coh_thresh: (0.7) coherence threshold
    - freq_av: (False) activate/deactivate band averaging
    - ppd: (10) frequency points per decade, if band averaging is enabled
    
    OUTPUTS:
    - fk (frequency axis)
    - squared coherence (purely real)
    - transfer function (complex)
    
    
    Last updated: 15/1/2014
    Author of last update: Roberto Calandrini    
    
    """
    
    optparams={
               'glob_rem_DC': True,
               'glob_detrend': False,
               'loc_rem_DC': True,
               'loc_detrend': True,
               'calibration1': None,
               'calibration2': None,
               'robust': False,
               'ppd': 10,
               'coh_thresh': 0.7,
               'freq_av': False,
               'verbose': False,
               'experimental': False,
               }
    
    optparams.update(kwargs)
    
    #If no calibration is available, just put ones
    if optparams['calibration1']==None:
        optparams['calibration1']= np.ones(M/2., dtype=np.double)+1j*np.zeros(M/2., dtype=np.double)
        
    if optparams['calibration2']==None:
        optparams['calibration2']= np.ones(M/2., dtype=np.double)+1j*np.zeros(M/2., dtype=np.double)
    
    #Create positive frequency axis removing DC
    if( np.mod(M,2)==0 ):
        fk= np.fft.fftfreq( int(M), 1./fc)[:int(M)/2+1]
        fk= fk[1:]
        fk[-1]= -fk[-1] #make Nyquist positive
    else:
        M= 2**(np.ceil(np.log2(M))) #convert to the next pow of 2
        print "The window length should be a power of 2 to obtain best performance during FFT processing.\nThe new window length used is: "+str(M)
        fk= np.fft.fftfreq( int(M), 1./fc)[:int(M)/2+1]
        fk= fk[1:]
        fk[-1]= -fk[-1] #make Nyquist positive
    
    # Prepare temporal segments
    lowmem= False
    try:
        [sig1,L1]= windowing(signal1,M,over,wind, glob_rem_DC= optparams['glob_rem_DC'], glob_detrend=optparams['glob_detrend'],
                            loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
        [sig2,L2]= windowing(signal2,M,over,wind, glob_rem_DC= optparams['glob_rem_DC'], glob_detrend=optparams['glob_detrend'],
                        loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
    except MemoryError:
        L1,_= lowmem_prepare(signal1,M,over,wind, glob_rem_DC= optparams['glob_rem_DC'], glob_detrend=optparams['glob_detrend'],
                           loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
        L2,R= lowmem_prepare(signal2,M,over,wind, glob_rem_DC= optparams['glob_rem_DC'], glob_detrend=optparams['glob_detrend'],
                           loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
        lowmem= True
    
    #Cut to the minimum number of slices -> the shortest signal in time win
    L= np.min([L1,L2])
    
    if optparams['verbose']:
        print "\nNumber of subsegments extracted: "+str(L)
        print "\nVariance reduction to expect: < "+str( np.sqrt(L) )
    
    if not lowmem:
        #Created slice generators
        sgen1= spectraGen(sig1,L)
        sgen2= spectraGen(sig2,L)
    else:
        sgen1= spectraGen_lowmem(signal1,L,R,wind, loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
        sgen2= spectraGen_lowmem(signal2,L,R,wind, loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
    
    #Initialize memory for spectra
    CRspectrum = np.zeros_like(fk, dtype= np.complex128) #real part of cross spectra
    Pspectra1 = np.zeros_like(fk, dtype= np.float64)
    Pspectra2 = np.zeros_like(fk, dtype= np.float64)
    
    if optparams['experimental']:
        CRspectrum_e= np.zeros_like(fk, dtype= np.complex128)
    
    #Normalization factor to compensate for the window
    #Correct for continuous spectrum (no line components)
    win_en2= np.sum(wind**2)
    if optparams['robust']:
        ncount=0
    i=0
    while(i<L):
        # Extract signals segments
        temp1= next(sgen1)
        temp2= next(sgen2)
        
        # Robustness: coherence thresholding
        if optparams['robust']:
                
                # Averaging in frequency [Daniell method]
                # Much faster, same results.. it seems.
                # Whitening of Cross and power spectra before summing
                tf1= np.fft.rfft(temp1)[1:]/(optparams['calibration1'])
                tf2= np.fft.rfft(temp2)[1:]/(optparams['calibration2'])
                
                CRsp= (tf1*np.conj(tf2))
                Psp1= np.abs(tf1)**2 
                Psp2= np.abs(tf2)**2 
                
                Coh2= np.abs(np.sum(CRsp))**2/(np.sum(Psp1)*np.sum(Psp2))
                if(Coh2<optparams['coh_thresh']):
                    i +=1
                    continue
                
                # No recalculation
                corspe1= tf1
                corspe2= tf2
        else:
                #Calculate calibrated spectra
                corspe1= np.fft.rfft( temp1 )[1:] /(optparams['calibration1'])
                corspe2= np.fft.rfft( temp2 )[1:] /(optparams['calibration2'])
            
        #Apply NAN masking
        mask1= np.isnan(corspe1)
        mask2= np.isnan(corspe2)
        corspe1[mask1]= 0.0 +1j*0.0
        corspe2[mask2]= 0.0 +1j*0.0
        
        #Accumulators
        Pspectra1 += ( np.abs( corspe1 )**2 ) #/ (fc*win_en2)
        Pspectra2 += ( np.abs( corspe2 )**2 ) #/ (fc*win_en2)
        CRspectrum += ( corspe1* np.conj( corspe2 ) ) # S1*S2^h; /(fc*win_en2)
        
        # Secondary estimate: experimental
        if optparams['experimental']:
            CRspectrum_e += ( corspe2* np.conj( corspe1 ) )/(win_en2*fc)
            
        if optparams['robust']:
            ncount += 1
        i+=1
    
    # Stacked version
    # test the denominator: if 0/0 --> NaN
    if optparams['freq_av']:
        # Band averaging [Fede]
        nf1= 5                      # first freq index used
        nfend= np.floor((M/2)*0.8)  # last freq index used
        NFPD= optparams['ppd']                    # frequencies per decade
        outfreqaxis, baindices= SetBandAveraging(fk, nf1, nfend, NFPD)
        
        nfreq = len(outfreqaxis)
        CRspectrum_bavg = np.zeros(nfreq, dtype=np.complex128)
        Pspectra1_bavg= np.zeros(nfreq, dtype= np.float64)
        Pspectra2_bavg= np.zeros(nfreq, dtype= np.float64)
        if optparams['experimental']:
            CRspectrum_e_bavg = np.zeros(nfreq, dtype=np.complex128)
        for ifreq in range(nfreq):
            CRspectrum_bavg[ifreq] = np.mean(CRspectrum[baindices[ifreq]],axis=0)
            if optparams['experimental']:
                CRspectrum_e_bavg[ifreq] = np.mean(CRspectrum_e[baindices[ifreq]],axis=0)
                
            Pspectra1_bavg[ifreq] = np.mean(Pspectra1[baindices[ifreq]],axis=0)
            Pspectra2_bavg[ifreq] = np.mean(Pspectra2[baindices[ifreq]],axis=0)
        
        # Initialize output variables
        Coh2_l= np.zeros(nfreq, dtype= np.float64)
        TF_l= np.zeros(nfreq, dtype= np.complex128)
        fk= outfreqaxis
        
        maskk= np.logical_and( Pspectra1_bavg!=0.0,Pspectra2_bavg!=0.0 )
        Coh2_l[maskk]= np.abs(CRspectrum_bavg[maskk])**2 / (Pspectra1_bavg[maskk]*Pspectra2_bavg[maskk])
        Coh2_l[np.logical_not(maskk)]= np.nan
        
        #Two way to calculate the TF
        if optparams['experimental']:
            maskk2= np.logical_and(Pspectra2_bavg!=0.0, np.abs(CRspectrum_e_bavg)!=0.0)
            TF_l[maskk2]= ( ( CRspectrum_bavg[maskk2] * CRspectrum_e_bavg[maskk2] ) + ( Pspectra2_bavg[maskk2]*Pspectra1_bavg[maskk2] ) ) / (2.*Pspectra2_bavg[maskk2]*CRspectrum_e_bavg[maskk2])
            TF_l[np.logical_not(maskk2)]= np.nan
        else:
            maskk2= Pspectra2_bavg!=0.0
            TF_l[maskk2]= CRspectrum_bavg[maskk2]/ Pspectra2_bavg[maskk2]
            TF_l[np.logical_not(maskk2)]= np.nan
        
    else:
        # Initialize output variables
        Coh2_l= np.zeros_like(fk, dtype= np.float64)
        TF_l= np.zeros_like(fk, dtype= np.complex128)
        
        # No band averaging
        maskk= np.logical_and( Pspectra1!=0.0,Pspectra2!=0.0 )
        Coh2_l[maskk]= np.abs(CRspectrum[maskk])**2 / (Pspectra1[maskk]*Pspectra2[maskk])
        Coh2_l[np.logical_not(maskk)]= np.nan
        
        #Two way to calculate the TF
        if optparams['experimental']:
            maskk2= np.logical_and(Pspectra2!=0.0, np.abs(CRspectrum_e)!=0.0)
            TF_l[maskk2]= ( ( CRspectrum[maskk2] * CRspectrum_e[maskk2] ) + ( Pspectra2[maskk2]*Pspectra1[maskk2] ) ) / (2.*Pspectra2[maskk2]*CRspectrum_e[maskk2])
            TF_l[np.logical_not(maskk2)]= np.nan
        else:
            maskk2= Pspectra2!=0.0
            TF_l[maskk2]= CRspectrum[maskk2]/ Pspectra2[maskk2]
            TF_l[np.logical_not(maskk2)]= np.nan
        
    
    #Global frequency averaging, if active
        
    
    if optparams['robust']:
        if ncount==0:
            raise Exception("\nNo subsegment has passed coherence thresholding: lower the threshold (actual: "+str(optparams['coh_thresh'])+") or use non robust version")
        else:
            print "\n"+str(ncount)+" segment have passed the coherence thresholding test, and so are used to calculate TF and coherence"
    
    return fk, Coh2_l, TF_l
    
    
def cleanOBS(data,plot=False, label="Data"):
    """
    Look for anomalous data, outliers, and do a substitution through interpolation
    """
    import matplotlib.pyplot as plt
    from scipy import interpolate
    
    #exclude bad data points through threshold
    N= data.size
    x= np.arange(N)    
    datan= data.copy()
    mu= np.mean(datan)
    ddata= datan-mu
    thresh= 3*np.std(ddata)
    
    if plot:
        fig= plt.figure()
        fig.suptitle(label+"- Data +- threshold")
        ax= fig.add_subplot(111)
        ax.plot(x,ddata,'b-'),ax.grid(True)
        ax.plot(x,np.ones_like(x)*thresh,'r')
        ax.plot(x,-np.ones_like(x)*thresh,'r')
        
        plt.show()
        plt.close(fig)
        
    # Masking
    mask_good= (ddata<thresh) & (ddata>-thresh)
    
    if plot:
        fig= plt.figure()
        fig.suptitle(label+" cleaned")
        ax= fig.add_subplot(111)
        ax.plot(x[mask_good],ddata[mask_good],'b-'),ax.grid(True)
        
        plt.show()
        plt.close(fig)
        
    # Interpolation
    datab= interpolate.griddata(x[mask_good],ddata[mask_good],x)
    
    if plot:
        fig= plt.figure()
        fig.suptitle(label+" clean + regrid")
        ax= fig.add_subplot(111)
        ax.plot(x,datab,'b-'),ax.grid(True)
        
        plt.show()    
        plt.close(fig)
        
    datab += mu
    
    if plot:
        fig= plt.figure()
        fig.suptitle(label+" residual")
        ax= fig.add_subplot(111)
        ax.plot(x,np.abs(datab-datan),'m-'),ax.grid(True)
        
        plt.show() 
        plt.close(fig)
        
    return datab
    
    
def CohTempVar(signal1, signal2, fc, M, over, wind, **kwargs):
    
    optparams={
               'glob_rem_DC': True,
               'glob_detrend': False,
               'loc_rem_DC': True,
               'loc_detrend': True,
               'calibration1': None,
               'calibration2': None,
               'robust': False,
               'ppd': 10,
               'coh_thresh': 0.7,
               'freq_av': False,
               'verbose': False,
               'experimental': False,
               'startTime': 0,
               'typed': 'coherence',
               'label1': 'Site1',
               'label2': 'Site2',
               'label_ch1': 'Ch1',
               'label_ch2': 'Ch2',
               'label_r1': 'run0',
               'label_r2': 'run0',
               'plot': False,
               'smooth': 10,
               'Nindex': 10,
               'cohgram_type': 'index', # norm/index
               }
    
    optparams.update(kwargs)
    
    from matplotlib.dates import date2num
    import datetime
    import matplotlib.pyplot as plt
    import pytz
    from matplotlib.dates import DateFormatter, AutoDateLocator
    import matplotlib.ticker
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.colors as col
    import matplotlib.cm as cm
    
    
    #If no calibration is available, just put ones
    if optparams['calibration1']==None:
        optparams['calibration1']= np.ones(M/2., dtype=np.double)+1j*np.zeros(M/2., dtype=np.double)
        
    if optparams['calibration2']==None:
        optparams['calibration2']= np.ones(M/2., dtype=np.double)+1j*np.zeros(M/2., dtype=np.double)
    
    #Create positive frequency axis removing DC
    if( np.mod(M,2)==0 ):
        fk= np.fft.fftfreq( int(M), 1./fc)[:int(M)/2+1]
        fk= fk[1:]
        fk[-1]= -fk[-1] #make Nyquist positive
    else:
        M= 2**(np.ceil(np.log2(M))) #convert to the next pow of 2
        print "The window length should be a power of 2 to obtain best performance during FFT processing.\nThe new window length used is: "+str(M)
        fk= np.fft.fftfreq( int(M), 1./fc)[:int(M)/2+1]
        fk= fk[1:]
        fk[-1]= -fk[-1] #make Nyquist positive
    
    # complete spectral matrix for each channel
    [sig1,L1]= windowing(signal1,M,over,wind, glob_rem_DC= optparams['glob_rem_DC'], glob_detrend=optparams['glob_detrend'],
                            loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'], transform=True)
    [sig2,L2]= windowing(signal2,M,over,wind, glob_rem_DC= optparams['glob_rem_DC'], glob_detrend=optparams['glob_detrend'],
                    loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'], transform=True)
                    
    
    # NUMERICAL PART
    
    coh_t= np.zeros( L1, dtype=np.double )
    coh_f= np.zeros( M/2, dtype= np.double )
    cohgram= np.zeros( (M/2,L1), dtype= np.double )
    
     # Coherence calculated with Daniell method for each segment-stack in FD
    for i in range( int(L1) ):
        sp1= np.abs(sig1[:,i])**2
        sp2= np.abs(sig2[:,i])**2
        csp= sig1[:,i] * np.conj(sig2[:,i])
        
        coh_t[i]= np.abs( np.sum(csp) )**2/( np.sum(sp1)*np.sum(sp2) )
    
    # Classic coherence - stack in TD
    # To produce a coherogram - smooth the cross spectrum in time
    Lw= float(optparams['smooth'])
    
    # smoother
    ww= np.hamming(Lw)
    ww *= (1./np.sum(ww))
    
    for i in range( int(M/2) ):
        sp1= np.abs(sig1[i,:])**2
        sp2= np.abs(sig2[i,:])**2
        csp= sig1[i,:] * np.conj( sig2[i,:] )
        
        # stacked
        coh_f[i]= np.abs( np.sum(csp) )**2/( np.sum(sp1)*np.sum(sp2) )
        
        # coherogram - smoothed
        cohgram[i,:]= np.abs( np.convolve(csp,ww,mode='same') )**2/( np.convolve(sp1,ww,mode='same')*np.convolve(sp2,ww,mode='same') )
        
    # Now test if there's local correlation btw 2 measures at the same frequency
    ttax_w= np.linspace(optparams['startTime'],optparams['startTime']+signal1.size/fc, L1)
    # convert in matplotlib datetime format
    dtime_w= date2num([datetime.datetime.fromtimestamp(float(_t), tz=pytz.utc) for _t in ttax_w])
    
    xlocator = AutoDateLocator(tz=pytz.utc, minticks=2, maxticks=8, interval_multiples=True)
    dateFormatter = DateFormatter('%d/%m %H:%M:%S', tz=pytz.utc)    
    
    run1= str(datetime.datetime.fromtimestamp( float(optparams['label_r1'].split("_")[0]) ) )
    run2= str(datetime.datetime.fromtimestamp( float(optparams['label_r2'].split("_")[0]) ) )
    
    if optparams['typed']=='classic':
        # Plots
        fighh= plt.figure()
        ax= fighh.add_subplot(211)
        ax2= fighh.add_subplot(212)
        
        ax.set_title("Coherence variation with time - Stack in FD, sampleRate: "+str(fc)+", "+optparams['label1']+"_"+run1+"-"+optparams['label2']+"_"+run2+", channels: "+optparams['label_ch1']+"-"+optparams['label_ch2'] )
        ax.plot( dtime_w, coh_t, '*k-' ),ax.grid(True)
        ax.set_ylim(0,1.1)
        ax.tick_params(labelsize=10)
        ax.xaxis.set_major_locator(xlocator)
        ax.xaxis.set_major_formatter(dateFormatter)
        ax.set_xlabel("Datetime segment")
        
        ax2.set_title("Coherence variation with Period - Stack in TD, sampleRate: "+str(fc)+", "+optparams['label1']+"_"+run1+"-"+optparams['label2']+"_"+run2+", channels: "+optparams['label_ch1']+"-"+optparams['label_ch2'])
        ax2.semilogx( 1./fk, coh_f, '*b-' ),ax2.grid(True)
        ax2.set_ylim(0,1.1)
        ax2.set_xlabel("Periods [s]")
        
    else:
        # Coherogram
    
        fighh= plt.figure( str(datetime.datetime.now()), figsize=(12,12) )
        fighh.suptitle("Coherogram - Sites: "+optparams['label1']+"_"+run1+"-"+optparams['label2']+"_"+run2+", Channels: "+optparams['label_ch1']+"-"+optparams['label_ch2']+", sampleRate: "+str(fc) )
        axc= fighh.add_axes([0.05,0.1,0.8,0.8])
        
        #Create indexed colormap
        Nc= optparams['Nindex']+1
        
        if optparams['cohgram_type']=='norm':
            cmap= cm.get_cmap(name='jet')
            norm= col.Normalize()
        else:
            cmap= cm.get_cmap(name='jet',lut=Nc)
            norm= col.BoundaryNorm( np.linspace(0.0,1.0,Nc),Nc)
        
            
        hmesh = axc.pcolormesh( dtime_w,1./fk, cohgram, vmin=0,vmax=1.0, cmap=cmap, norm=norm,
                               rasterized=True);
                               
        #reserve space for colorbar
        divider = make_axes_locatable(axc)
        cax = divider.append_axes("right", size="2%", pad=0.25)
        
        #colorbar
        hcb = plt.colorbar(hmesh,cax=cax,ticks=matplotlib.ticker.MaxNLocator(Nc-1),orientation='vertical')
        hcb.ax.tick_params(labelsize=12, direction='out')
        
        #set axis limits
        axc.set_xlim(min(dtime_w), max(dtime_w))
        axc.set_ylim( 10**min( np.log10(1./fk) ), 10**max( np.log10(1./fk) ))
        
        #set axis scale        
        axc.set_yscale('log')
        axc.invert_yaxis()
        
        #Set axis labels and ticks
        #ax.set_ylabel(self.channel, fontsize=14)
        axc.tick_params(labelsize=10)
        axc.xaxis.set_major_locator(xlocator)
        axc.xaxis.set_major_formatter(dateFormatter)
        
        #ax.axis('tight')
        axc.set_ylabel('Periods [s]')
                               
        # Location of the coherence zone in time-frequency space >= coh_thresh and delimit
        mask_coh= cohgram>=optparams['coh_thresh']
        
        if np.logical_not(mask_coh).all():
            print "\nApplying the coherence thresholding left no points in TF domain: continue without thresholding..."
            
            if optparams['plot']:
                plt.show()
                
            return fighh, -999, -999, -999, -999
        else:
            # ZOI definition
            
            # Saturation of the zone of interest (off at the moment)
#            cohgram[mask_coh]= 1.1
#            hmesh2 = axc.pcolormesh( dtime_w,1./fk, cohgram, vmin=0,vmax=1.0, cmap=cmap, norm=norm,
#                                   rasterized=False);
#            hmesh2.set_alpha(0.7)
            
            # To decide which temporal segment is the best, compress over FD - Daniell method
            # then threshold with the mean
            # Do also compression over TD, to select best frequency band in which coh>=coh_thresh
            comp_mask_TD= np.sum(mask_coh,0)
            
            # Operating directly on the mask will mantain the 'y' axis in linear frequency!
            comp_mask_FD= np.sum(mask_coh,1)
            
            # Mean for threshold must be calculated excluding zeroes
            m_valid_TD= np.logical_not( comp_mask_TD==0 )
            m_valid_FD= np.logical_not( comp_mask_FD==0 )
            m_TD= comp_mask_TD>=np.mean(comp_mask_TD[m_valid_TD])
            m_FD= comp_mask_FD>=np.mean(comp_mask_FD[m_valid_FD])
            
            # Get temporal and frequency indexes
            start_TD= ttax_w[m_TD][0]
            end_TD= ttax_w[m_TD][-1]
            
            for ind,i in enumerate(ttax_w):
                if i==start_TD:
                    i_start_TD= ind
                elif i==end_TD:
                    i_end_TD= ind
                    
            # Samples in FD are localized in linear frequency
            start_FD= fk[m_FD][0]
            end_FD= fk[m_FD][-1]
            
            for ind,i in enumerate(fk):
                if i==start_FD:
                    i_start_FD= ind
                elif i==end_FD:
                    i_end_FD= ind
                    
            # Overplot the limits on the coherogram plot
#            axc.plot( start_TD*np.ones(len(fk)), (1./fk), 'k-', linewidth=7 )
#            axc.plot( end_TD*np.ones(len(fk)), (1./fk), 'k-', linewidth=7 )
#            
#            axc.semilogy( dtime_w, (1./start_FD)*np.ones(len(dtime_w)), 'k-', linewidth=7 )
#            axc.semilogy( dtime_w, (1./end_FD)*np.ones(len(dtime_w)), 'k-', linewidth=7 )
        
            if optparams['plot']:
                plt.show()
                
            return fighh, start_TD, end_TD, start_FD, end_FD
    
    

def DCdetrend(signal):
    import scipy.signal as scisi
    
    signal -= np.mean(signal)
    signal= scisi.detrend(signal)
    
    return signal

def spectrogram(signal, M, over, wind, fc, **kwargs ):
    """
    Calculates the spectrogram of the input signal
    
    Default to power spectrum ['p' type]
    It's possible to select also 'a', for amplitude auto-spectral density
    
    The routine INPUTS are (ordered):
    - signal
    - M (window length)
    - over (window overlap)
    - wind (window used, must be of length M)
    - fc (Sampling rate)
    
    Optional args:
    - glob_rem_DC (boolean, remove DC from the whole signal)
    - glob_detrend (boolean,remove trends from the whole signal)
    - loc_rem_DC (boolean, remove DC from local slice)
    - loc_detrend (boolean, remove trends from local slice)
    - type ('p' or 'a', type of spectra of interest: power or amplitude)
    - calibration (default to False, put in calibration function if you want the input to be calibrated)
    - verbose (boolean, show more diagnostic messages)
    - plot (boolean, default to False)
    - index_plot ('T' or 'F', default to time; let the user navigate through the spectrogram object slicing it in time or frequency)
    
    OUTPUTS:
    - fk, TW, Spectrogram matrix: respectively returns the frequency axis, the time windows axis and the spectrogram
    
    NB:  the routine automatically removes DC component
    """
    optparams={
               'glob_rem_DC': True,
               'glob_detrend': False,
               'loc_rem_DC': True,
               'loc_detrend': True,
               'type': 'p',
               'calibration': None,
               'verbose': False,
               'plot': True,
               'plot_type': 'full',
               'index_plot': 'T',
               'start_time': 0
               }
    
    optparams.update(kwargs)
    
    #Windowing
    [sig,L]= windowing(signal,M,over,wind, glob_rem_DC= optparams['glob_rem_DC'], glob_detrend=optparams['glob_detrend'],
                       loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
    
    #Create positive frequency axis removing DC
    fk= np.fft.fftfreq(M, 1./fc)[:M/2+1]
    fk= fk[1:]
    fk[-1]= -fk[-1] #make Nyquist positive
    
    #time windows axis
    Tw= np.arange(0,L*(M*(1./fc)),(M*(1./fc)))
    
    if optparams['verbose']:
        print "\nNumber of subsegments extracted: "+str(L)
        print "\nVariance reduction to expect: < "+str( np.sqrt(L) )
    
    if optparams['type']=='p' or optparams['type']=='a':
        sgen= spectraGen(sig,L)
        Pspectrum= np.zeros( (M/2, L) )
        win_en2= np.sum(wind**2)
        if optparams['calibration']==None:
            ind=0
            for i in sgen:
                Pspectrum[:,ind]= ( np.abs( np.fft.rfft(i)[1:] )**2 / (fc*win_en2) )
                ind+=1
        else:
            ind=0
            for i in sgen:
                Pspectrum[:,ind]= ( np.abs( np.fft.rfft(i)[1:] /(optparams['calibration']) )**2 / (fc*win_en2) )
                ind+=1
                
    if optparams['plot_type']=='slice':
        fig=plt.figure( figsize=(12,12) )
        ax=fig.add_subplot(111)
        
        if optparams['index_plot']=='T':
            tracker = IndexTracker(ax, Pspectrum,fk, index='T')
        else:
            tracker = IndexTracker(ax, Pspectrum,Tw, index='F')
            
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)

        plt.show()
        
    else:
        #to be implemented
        pass
        
    return fk,Tw,Pspectrum


def indices(a, func):
    """
    Return indices of the positive results returning from a test function passed as input "func": this must be a callable         
    """ 
    return [i for (i, val) in enumerate(a) if func(val)] 

     
def set_bands(fk, **kwargs):
    """
    Compute center frequencies and averaging bands expressed as FFT bin indexes to do frequency averaging using 
    a sequence of windows logarithmically and regularly spaced in the periods domain
    
    Expects a positive frequency semi-axis as INPUT
    
    INPUTS:
        in_sm: window half-width at start
        inc: window size increase factor in percentage
        plot: boolean toggle to visualize the window sequence
    OUTPUTS:
        f_center: bands central frequency
        bands_index: FFT bin indexes that define each band [max_bound,min_bound]
    
    NB: Mantained for retro-compatibility with adu_database functions: pstselftest and plotasd
    """
    
    optparams={
               'hwidth_start': 0.05,
               'inc': 0.15,
               'decade_p': 10,
               'plot': False,
               }
    
    optparams.update(kwargs)
    
    #Creation of the logarithmically spaced period vector
    st= float( optparams['decade_p'] )
    T=1.0/fk #discrete periods [s] -> discard DC
    df= fk[0] #frequency resolution
    maxT= np.log10(T[0])  #longest period [s]
    minT= np.log10(T[-1]) #shortest period [s]
    step= 1./st #points per decade sampling
    warp=np.arange(minT,maxT,step) #from LOW to HI frequency (from LONG to SHORT periods) --> this way the high frequency range has a finer sampling
    T_center= 10.0**(warp) #logaritmically equidistant periods
    
    # There will be no DC
    f_center= 1.0/T_center #logaritmically equidistant frequencies, with an hi frequency denser sampling
    NN= f_center.size
    
    #mapping/lock to frequency bins
    for i,val in enumerate(f_center):
        f_center[i]= fk[np.argmin( np.abs(fk-val) )]

                
    #Bands calculation
    tol= optparams['hwidth_start'] #starting window half-width
    inc= optparams['inc'] #windows size increase factor
    
    #contains the lower and upper frequency bound for each f_center   
    bands=np.zeros( (2,f_center.size), dtype=np.double )
    
    
    i=0
    while i<NN:
        val= f_center[i]
        maxf= np.ceil( (val*(1.0+1.5*tol))/df )*df #max frequency bound of the averaging window
        minf= np.floor( (val*(1.0-0.5*tol))/df )*df #min frequency bound of the averaging window
        
        tol *= (1+inc) #increase the width of the averaging window at each step
        if maxf < minf:
            maxf= minf
            minf= maxf
        
        if ( maxf<= np.max(fk)  and  minf>=df ):
            #need to be sure that the bounds are included in the frequency axis
            bands[:,i]=[maxf,minf]
            i +=1
        elif( maxf>np.max(fk) and minf>=df ):
            #if the maximum bound exceeds, lock it to max
            bands[:,i]=[np.max(fk),minf]
            i +=1
        elif( maxf<= np.max(fk)  and  minf<df ):
            #if the minimum bound exceeds, lock it to min
            bands[:,i]=[maxf,df]
            i +=1
        else:
            #if the bounds are completely off, lock to the frequency bounds of the axis
            bands[:,i]=[np.max(fk),df]
            i +=1

    #return to indices
    bands_index= bands/df
    f_index= f_center/df
    
    if optparams['plot']:
        
        fig= plt.figure()
        ax= fig.add_subplot(211) 
        ax.loglog( f_center, bands[1,:], 'b*-' ), ax.grid(True), ax.hold(True),
        ax.loglog( f_center, bands[0,:], 'r*-' ),
        ax.loglog( f_center, f_center, 'k-' ),
        ax.set_xlabel("Frequency [Hz]"),ax.set_ylabel("Band bounds")
        ax.legend(['Start frequency','End frequency'],loc='best')
        ax1= fig.add_subplot(212)
        ax1.loglog( f_index, bands_index[1,:], 'b*-'),ax1.grid(True), ax1.hold(True),
        ax1.loglog( f_index, bands_index[0,:], 'r*-' ),
        ax1.loglog( f_index, f_index, 'k-' ),
        ax1.set_xlabel("Frequency [Hz]"),ax1.set_ylabel("Band bounds indexes")
        ax1.legend(['Start frequency bin','End frequency bin'],loc='best')
        
        plt.show()
        
    return f_center,bands_index

    
def calc_frsp(M,fc,filen, **kwargs):
    """
    Read calibration file and calculate frequency response of the 
    instrument, interpolated on the signal frequency grid
    """
    
    optparams={
               'remDC': True,
               'plot': False,
               'noc': False, #no calibration flag, used in ADU_Console: PSTselftest and ASDtest
               }
    
    optparams.update(kwargs)
    
    #create positive frequency axis & set positive Nyquist frequency, automatically excluding DC
    df= fc/ float(M)
    if(M%2==0):
        NN= M/2.
        if optparams['remDC']:
            fk= np.arange(1,NN+1,1)*df # --> automatic exclusion of DC
        else:
            fk= np.arange(0,NN+1,1)*df # --> automatic exclusion of DC
    else:
        NN= ((M-1)/2.)
        if optparams['remDC']:
            fk= np.arange(1,NN+1,1)*df # --> automatic exclusion of DC
        else:
            fk= np.arange(0,NN+1,1)*df # --> automatic exclusion of DC
    try:
        RSP= GEN_interp(filen,fk,plot=optparams['plot'])
        
        if optparams['noc']:
            noc= True
            
            return RSP,fk,noc
        else:            
            return RSP,fk
            
    except Exception as exc:
        print "\nException: "+exc.message
        print "\nWarning: calibration file not found, continue without calibration.." + filen
        
        RSP= np.ones_like(fk)+1j*np.zeros_like(fk)
        
        if optparams['noc']:
            noc= False
            
            return RSP,fk,noc
        else:            
            return RSP,fk
    

def GEN_interp(GENfile,fk,**kwargs):
    optparams={
               'plot': False,
               
               }
     
    import MT_Utils.ConvertCalibration as convert 
    
    optparams.update(kwargs)
    
    #Loading the entire Calibration file
    freq,amp,phase,name = convert.ReadGeneralRSP(GENfile)
    phase= np.deg2rad(phase) #convert to [rad]
    
    #At this point I already know the choppper's state
    
    #Interpolation of the calibration response on the frequency grid of the signal [positive frequency semi-axis; 0-> Nyquist]
    new_amp = 10.**(np.interp( np.log10(fk), np.log10(freq), np.log10(amp) , left= np.NAN, right= np.NAN) )
    
    new_phase = np.interp(np.log10(fk), np.log10(freq), phase, left= np.NAN, right= np.NAN)        
    
    #Return complex response on the positive frequency axis
    RSP= new_amp*np.exp(1j*new_phase)
    
    
    if(optparams['plot']):
        argfreq= np.argsort(freq)
        fig=plt.figure('Coil Frequency Response '+name.strip('.GEN') )
        ax=fig.add_subplot(2,1,1)
        ax.loglog(fk,new_amp,'k-*', zorder=0), ax.grid(True)
        ax.hold(True), ax.loglog(freq[argfreq],amp[argfreq],'r*', markersize=12, zorder=2)
        ax.set_xlabel('Frequency [Hz]'), ax.set_ylabel('Amplitude')
        ax.legend(['Interpolated','Original'])
        
        ax2=fig.add_subplot(2,1,2)
        ax2.semilogx(fk,np.rad2deg(new_phase),'k-*', zorder=0), ax2.grid(True)
        ax2.hold(True), ax2.semilogx(freq[argfreq],np.rad2deg(phase[argfreq]),'r*', markersize=12, zorder=2)
        ax2.set_xlabel('Frequency [Hz]'), ax2.set_ylabel('Phase [deg]')
        ax2.legend(['Interpolated','Original'])
        
        plt.show()
        
    
    return RSP
    

def TXT2RSP(TXTFile,fk,chop,**kwargs):
    """
    This function convert a calibration file of the Metronix's format, from TXT to 
    a vector that contains the frequency response of the instruments resampled on 
    the frequency axis fk given as an input; it takes into account whether the chopper 
    option is True or False
    """
    
    optparams={
               'plot': False,
               
               }
    
    import MT_Utils.ConvertCalibration as convert 
    
    optparams.update(kwargs)
        
    #Loading the entire Calibration file
    CalibrationData = convert.ReadCalibrationTXT(TXTFile)
        
    #Check the chopper's status
    if(chop==1):
        chop='On'
    else:
        chop='Off'

    #Select which frequency response is to retrieve, based on chopper flag value
    txtfreq = CalibrationData[chop][:,0]
    txtamp = CalibrationData[chop][:,1]
    txtphase = np.unwrap(CalibrationData[chop][:,2]*np.pi/180.) # [rad]
    
    
    #Interpolation of the calibration response on the frequency grid of the signal [positive frequency semi-axis; 0-> Nyquist]
    amp = 10.**(np.interp( np.log10(fk), np.log10(txtfreq), np.log10(txtamp) ) )
    amp=amp*fk
    
    phase = np.interp(np.log10(fk), np.log10(txtfreq), txtphase)        
    
    #Return complex response on the positive frequency axis
    RSP= amp*np.exp(1j*phase)
    
    if(optparams['plot']):
        argfreq= np.argsort(txtfreq)
        argf= np.argsort(fk)
        fig=plt.figure('Coil Frequency Response')
        ax=fig.add_subplot(2,1,1)
        ax.loglog(fk[argf],amp[argf],'k-*', zorder=0), ax.grid(True)
        ax.hold(True), ax.loglog(txtfreq[argfreq],txtamp[argfreq]*txtfreq[argfreq],'r*', markersize=12, zorder=2)
        ax.set_xlabel('Frequency [Hz]'), ax.set_ylabel('Amplitude')
        ax.legend(['Interpolated','Original'])
        
        ax2=fig.add_subplot(2,1,2)
        ax2.semilogx(fk[argf],np.rad2deg(phase[argf]),'k-*', zorder=0), ax2.grid(True)
        ax2.hold(True), ax2.semilogx(txtfreq[argfreq],np.rad2deg(txtphase[argfreq]),'r*', markersize=12, zorder=2)
        ax2.set_xlabel('Frequency [Hz]'), ax2.set_ylabel('Phase [deg]')
        ax2.legend(['Interpolated','Original'])
        
        plt.show()
        
    
    return RSP,np.max(txtfreq),np.min(txtfreq)
    
    
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def xcorr_f(sig1, sig2):
    """
    Cross-correlation in frequency domain
    S1= output of the signal model
    S2= input of the signal model
    """
    
    N1= sig1.size
    N2= sig2.size
    
    Nmax= N1+N2-1 #exact cross correlation length
    
    #round to next power of 2
    Nfft= np.int( 2**(np.ceil(np.log2(Nmax))) )
    
    S1c= np.conj( np.fft.rfft(sig1,n=Nfft) ) #INPUT
    S2= np.fft.rfft(sig2,n=Nfft) #OUTPUT
    
    #Get the cross-correlation
    result= np.real( np.fft.irfft( S1c*S2 ) )
    
    return result


def rotate(sig1,sig2,theta):
    """
    Assuming a right-handed reference cartesian system where sig1 points in the direction x and sig2 points in the direction y,
    rotates the system with a rotation angle theta, and return the rotated couple of signal in the new coordinates system (x1,y1)
    
    NB: expects theta in degrees
    """
    
    the= np.deg2rad(theta)
    sig1n= sig1*np.cos(the)-np.sin(the)*sig2
    sig2n= sig1*np.sin(the)+np.cos(the)*sig2
    
    return sig1n,sig2n


def cormat(signal1, signal2, M, wind, over, fc, **kwargs):
    """
    Use the polarized estimator to calculate the correlation 
    matrix of the observation
    
    Assumes that sig is an observation taken from a stationary ergodic process
    """
    
    optparams={
               'glob_rem_DC': True,
               'glob_detrend': False,
               'loc_rem_DC': True,
               'loc_detrend': True,
               'calibration1': None,
               'calibration2': None,
               }
    
    optparams.update(kwargs)
    
    R= np.zeros( 2*M-1, dtype=np.double )
    
    lowmem= False
    try:
        [sig1,L1]= windowing(signal1,M,over,wind, glob_rem_DC= optparams['glob_rem_DC'], glob_detrend=optparams['glob_detrend'],
                            loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
        [sig2,L2]= windowing(signal2,M,over,wind, glob_rem_DC= optparams['glob_rem_DC'], glob_detrend=optparams['glob_detrend'],
                        loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
    except MemoryError:
        L1,_= lowmem_prepare(signal1,M,over,wind, glob_rem_DC= optparams['glob_rem_DC'], glob_detrend=optparams['glob_detrend'],
                           loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
        L2,R= lowmem_prepare(signal2,M,over,wind, glob_rem_DC= optparams['glob_rem_DC'], glob_detrend=optparams['glob_detrend'],
                           loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
        lowmem= True
        
    #Cut to the minimum number of slices -> the shortest signal in time win
    L= np.min([L1,L2])
    
    if not lowmem:
        #Created slice generators
        sgen1= spectraGen(sig1,L)
        sgen2= spectraGen(sig2,L)
    else:
        sgen1= spectraGen_lowmem(signal1,L,R,wind, loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
        sgen2= spectraGen_lowmem(signal2,L,R,wind, loc_rem_DC= optparams['loc_rem_DC'], loc_detrend= optparams['loc_detrend'])
    
    #Pre-whitening of each segment involved in the cross-correlation
    i=0
    while(i<L):
        temp1= next(sgen1)
        temp2= next(sgen2)
        
        R+= (1./M)*np.convolve(temp1, temp2[::-1])
        i+=1
    
    R /= L
    
    #visual check --> OK
    #visual([R], type='s')
    
    #build corrmatrix
    CR= np.zeros( (M,M), dtype=np.double )
    for i in range(M):
        temp= R[i:i+M]
        CR[i,:]= temp[::-1]
        
    return CR

def coh_mat(Rx,Ry,Rxy):
    """
    Calculates coherence matrix
    """
    import matplotlib.cm as cm
    
    import scipy.linalg as la
    
    Rx12= la.cholesky(Rx)
    Ry12= la.cholesky(Ry)
    
    temp= np.dot( la.inv(Rx12), Rxy )
    cohm= np.dot( temp, la.inv(Ry12) )
    
#     check matrix
#     plt.imshow( np.log10(cohm) , cmap = cm.hot)
#     plt.colorbar()
#     plt.show()
    
    return cohm
    
def redrankcoh(Coh_mat, **kwargs):
    """
    Reduced rank coherence based on singular values
    thresholding
    """
    optparams={
               'th_mode': 'auto',
               'thresh': None,
               'get_sval': False,
               }
    
    optparams.update(kwargs)
    
    import scipy.linalg as la
    
    U,s,V= la.svd(Coh_mat)
    s= np.diag(s,0)
    
    if optparams['th_mode']=='auto':
        # Ok, set the threshold where there is the highest jump
        # in the singular values series
        ii= np.argmax( np.abs(np.diff(s)) )+100
    else:
        if optparams['thresh']!=None:
            ii= optparams['thresh']
        else:
            ii= np.ceil( np.size(s,0)/2. )
        
    
    Cmat_R=  np.dot( (np.dot( U[:,:ii+1], s[:ii+1,:ii+1])), V[:ii+1,:])
    
    if optparams['get_sval']:
        return Cmat_R,np.diag(s,0)
    else:
        return Cmat_R

def dftmat(L,K,fc, **kwargs):
    """
    Calculates DFT matrix, at equidistant
    pulsations on (0,2*pi)
    """
    
    optparams={
               'remDC': False,
               }
    
    optparams.update(kwargs)
    
    #discrete time axis
    n_ar= np.arange(0,L,1)
    
    #discrete frequency axis is handled separately
    if optparams['remDC']:
        k_ar= np.arange(1,K,1)
    else:
        k_ar= np.arange(0,K,1)
    #sampling just from 0 to Nyquist
    # this enters calculations
    ww= np.pi/K
    
    # for plotting
    df= (fc/2.)/K
    fk= k_ar*df
    
    #Calculation of DFT matrix
    if optparams['remDC']:
        DFT= np.zeros( (L,K-1), dtype=np.complex128 )
    else:
        DFT= np.zeros( (L,K), dtype=np.complex128 )
    
    for indk,k in enumerate(k_ar):
        DFT[:,indk]= (1./np.sqrt(L))*np.exp(1j*ww*n_ar*k)
        
    return fk,DFT
        