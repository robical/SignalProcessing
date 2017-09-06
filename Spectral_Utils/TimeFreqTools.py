
import numpy as np

def FFrFT(sig,a):
    """
    Implementation of the fast fractional fourier transform
    complexity is O(N*log(N))
    
    The fractional fourier transform represent the representation of 
    a signal on an oblique axis in the Wigner distribution domain (phase-space domain);
    this means it gives a representation which is intermediate between the time (or space) and 
    frequency (or wavenumber) domain.
    The application of an order alpha fractional fourier transform on a signal returns as an output 
    a signal which has a time-frequency representation rotated of an angle phi= alpha*(pi/2) in counter-clockwise
    direction.
    
    INPUT:
    signal= the signal under analysis
    a= the order of the fractional transform
    """
    
    signal = sig.copy()
    signal = signal.astype(np.complex)
    N= np.int32(signal.size)
    a= a % 4
    shift= np.remainder(np.arange(0,N,1, dtype=np.int32)+np.int32(np.fix(N/2)),np.ones(N, dtype=np.int32)*N)
    sN= np.sqrt(N)
    
    #special cases
    if (a==0): Fra= signal; return Fra
    if (a==1): Fra= np.fft.fft(signal[shift])*(1./sN); return Fra
    if (a==2): Fra= signal[::-1]; return Fra
    if (a==3): Fra= np.fft.ifft(signal[shift])*(np.float(N)/sN); return Fra
    
    #interval reduction for a: 0.5<a<1.5
    if (a>2.0): a -= 2; signal= signal[::-1]
    if (a>1.5): a -= 1; signal[shift]= np.fft.fft(signal[shift])*(1./sN);
    if (a<0.5): a += 1; signal[shift]= np.fft.ifft(signal[shift])*(np.float(N)/sN);
    
    #precompute parameters
    alpha= a*(np.pi/2.)
    s= np.pi/(4*np.sin(alpha)*(N+1))
    t= (np.pi/(N+1))*(np.tan(alpha/2.)/4.)
    Cs= np.sqrt(s/np.pi)* np.exp(-1j*(1-a)*(np.pi/4.))
    
    #sinc interpolation --> upsampling
    up_base= np.arange(-(2*N-3),(2*N-3)+2,2)
    sig1= fftconv(signal,np.sinc(up_base/2.),1)[N:2*N]
    
    #First chirp multiplication
    chirp= np.exp(-1j*t*(np.arange(-N+1,N+2,1)**2))
    #polyphase implementation
    lodd= chirp[1::2] #odd part
    leve= chirp[2::2] #even part
    
    signal= signal*lodd
    sig1= sig1*leve
    
    #Chirp convolution
    chirp= np.exp(1j*s*(np.arange(-(2*N-1),2*N,1))**2 )
    #polyphase implementation
    eodd= chirp[1::2]
    eeve= chirp[2::2]
    
    signal= fftconv(signal,eodd,0)
    sig1= fftconv(sig1,eeve,0)
    
    h= np.fft.ifft(signal+sig1)
    
    #Last chirp multiplication
    Fra= Cs*lodd*h[N:2*N]
    
    return Fra
    
    
def fftconv(sig1,sig2,d):
    """
    Implements convolution in frequency domain
    """
    
    N= sig1.size
    M= sig2.size
    cs= N+M-1
    Nfft= np.int32( 2**(np.ceil(np.log2(cs))) )
    res= np.zeros( cs, dtype= np.double )
    S1= np.fft.fft(sig1,Nfft)
    S2= np.fft.fft(sig2,Nfft)
    R= S1*S2
    if d>0:
        res= np.real(np.fft.ifft(R))[:cs]
        return res
    else:
        return R
    
def WDF(sig,fc, **kwargs):
    """
    Implementations of the Pseudo- Wigner-Ville time-frequency distribution
    (Pseudo == smoothing window on the each local autocovariance estimate)
    
    -> I've also added the possibility to stack in time many estimates, but there's still some strange 
    effect on the distribution when the stacking is severe (more than 10 sample)
    
    -> There's the possibility to improve the code, infact just an FFT on N samples is needed; now it's done on 2*N samples
    
    INPUTS:
    sig= signal under analysis
    fc= sampling rate
    
    OUTPUTS:
    wdf(i,k)= Wigner-Ville representation of the signal
    """
    import scipy.signal as scisi
    
    optparams={
               'dt': 1,
               
               }
    
    optparams.update(kwargs)
    
    dt_k= optparams['dt']
    
    #remove DC
    N= sig.size
    dt= 1./fc
    sig -= np.mean(sig)
    
    indexes= np.arange(0,N,dt_k)
    Nnew= indexes.size
    
    #filter data with modified hamming window
    #index vector
    i= np.arange(0,N,1)
    win= np.ones(N, dtype=np.double)
    mask_rise= np.logical_and(i>=0,i<=N/10.)
    mask_fall= np.logical_and(i>=(9./10.)*N,i<N)
    win[mask_rise]= 0.54 - 0.46* np.cos(10*np.pi*i[mask_rise]/N)
    win[mask_fall]= 0.54 - 0.46* np.cos(10*np.pi*(N-i[mask_fall])/N)
    
    sig *= win
    
    
    #Get analytic signal
    if np.all(np.isreal(sig)):
        sig_a= scisi.hilbert(sig,N)
    else:
        sig_a= sig
    
    #Reserve space for WDF
    nn= 2*N
    WDF= np.zeros( (Nnew,N+1), dtype=np.double )
    winn= np.abs(np.hamming(nn))
    winn /= np.sum(winn)
    
    aa=0
    for m in indexes:
        if aa>0:
            indx= np.arange(indexes[aa-1],indexes[aa],1)
            c= np.zeros(2*N, dtype= np.complex)
            for ll in indx:
                    c += local_cov(sig_a,ll,N,fc,plot=False)
        else:
            #Calculates local covariance
            c= local_cov(sig_a,m,N,fc,plot=False)
        
        C= np.fft.fft(c*(winn**2),nn)[:N+1]
        
        WDF[aa,:]= np.real( 2*dt*C )
        
        aa+=1
        
    return WDF

def local_cov(siga,m,N,fc, plot=False):
    """
    Calculates local covariance, that is then used in the 
    calculation of WV spectrum
    """
    dum= np.zeros(N, dtype= np.complex)
    ss= np.zeros(N, dtype= np.complex)
    c= np.zeros(2*N, dtype= np.complex)
    
    dt=1./fc
    coef= 2*dt
    
    #local autocorrelation
    i= np.arange(0,N,1)
    #index first part
    ii_1= m-i
    #index second component
    ii_2= m+i
    
    dum[ii_1>=0]= np.conj( siga[ii_1[ii_1>=0]] )
    ss[ii_2<N]= siga[ii_2[ii_2<N]]
    dum= coef*ss*dum
    c[:N]=dum
    i= N
    if (m>=i):
        c[N]= siga[m+i]*np.conj(siga[m-i])
        
    c[(N+1):]= np.conj(c[2:(N+1):][::-1])
    
    if plot:
        import matplotlib.pyplot as plt
        
        fig= plt.figure(figsize=(10,8))
        ax= fig.add_subplot(211)
        ax2= fig.add_subplot(212)
        ax.plot(np.real(c),'b-'),ax.grid(True),ax.plot(np.imag(c),'r-'),
        ax.set_ylim(-0.0002,0.0002)
        ax.legend(["Re","Im"])
        ax2.plot(np.abs(c),'k-'),ax2.grid(True),
        ax2.set_ylim(0,0.0006)
        ax2.legend(["Abs"])
        
        plt.show() 
    
    return c

def ist_freq(signal,fc,plot=False, **kwargs):
    """
    Instantaneous frequency estimation
    
    Following: Fomel, "Local seimic attributes"
    """
    
    import scipy.signal as scisi
    import scipy.linalg as lin
    
    optparams={
               'smoother': None,
               'smok': 3,
               'show_op': False,
               }
    
    optparams.update(kwargs)
    
    dt=1./fc
    N= signal.size
    
    #remove DC and detrend
    signal -= np.mean(signal)
    signal= scisi.detrend(signal)
    
    #Calculate analytic signal
    xa= np.imag( scisi.hilbert(signal) )
    
    ds= np.gradient(signal)
    dxa= np.gradient(xa)
    #Numerator
    num= (signal*dxa) - (ds*xa)
    den= signal**2 + xa**2
    
    D= np.diag(den,0)
    I= np.identity(N)
    
    # TO BE COMPLETED...
    if optparams['smoother']=='triangle':
        smok= optparams['smok']
        nsmok= (smok-1)/2
        weights= np.linspace(0,1,nsmok+2)[1::][::-1]
        for i in range(N):
            liminf= i-nsmok-1
            limsup= i+nsmok+1
            if liminf>=0 and limsup<=(N-1):
                I[i,i:limsup]= weights
                I[i,i:liminf]= weights
            elif liminf<0 and limsup<=(N-1):
                temp= np.arange(i,liminf,-1)
                limt= np.sum( np.int32(temp>=0) )
                I[i,i:limsup]= weights
                I[i,i:i-limt]= weights[:limt]
            else:
                temp= np.arange(i,limsup,1)
                limt= np.sum( np.int32(temp<=(N-1)) )
                I[i,i:limt]= weights[:limt]
                I[i,i:liminf]= weights

    #Resolve as a regularized inverse problem
    eps= 1e-3
    freq_ist= (fc/(2*np.pi))*( np.dot( lin.inv( (D+eps*I) ), num) ) 
    
    
    if plot:
        import matplotlib.pyplot as plt
        
        timeax= np.arange(0,N*dt,dt)
        
        fig= plt.figure(figsize=(10,8))
        ax= fig.add_subplot(111)
        ax.plot(timeax,freq_ist,'b-'),ax.grid(True),
        ax.set_xlabel("Time [s]"),
        ax.set_ylabel("Instant. frequency [Hz]"),
        
        if optparams['show_op']:
            fig2= plt.figure(figsize=(10,8))
            ax2= fig2.add_subplot(111)
            ax2.spy(I),
        
        plt.show()
    
    return freq_ist

def pscsdm(mdata,nfft,fc,output_site=None):
    """
    Calculates Pre-Stack Cross-spectral density matrix for Multivariate data
    
    mdata: dictionary that contain data and location for each of the site involved
    """
    import scipy.signal as scisi
    
    L= len(mdata) #Total number of trace
    
    if output_site!=None:
        Dlen= mdata[output_site]['data'].size
        for i in mdata.keys():
            if mdata[i]['data'].size<Dlen:
                Dlen= mdata[i]['data'].size
    else:
        Dlen= mdata[mdata.keys()[0]]['data'].size #Total number of samples for each trace
    
    #Calculate maximum number of section in the signal, with padding if necessary
    over= 0.5
    R= nfft*(1-over)
    Nwin= 1+ np.floor( (Dlen-nfft)/R )
    
    #Calculate window and window energy
    wind= np.hanning(nfft)
    we= np.sum(wind**2)
    Nwin= np.int32(Nwin)    
    
    #Reserve space
    csdm= np.zeros( (L,L,nfft/2,Nwin), dtype= np.complex128 )
    time_cache= np.zeros( (L,nfft), dtype=np.double )
    freq_cache= np.zeros( (L,nfft/2), dtype=np.complex128 )
    start=0
    stop= nfft
    for i in range(Nwin):
        #Fill timeseries cache
        for il,l in enumerate( sorted(mdata.keys()) ):
            time_cache[il,:]= mdata[l]['data'][start:stop]

            #Remove mean, detrend, window
            time_cache[il,:] -= np.mean(time_cache[il,:])
            time_cache[il,:] = scisi.detrend(time_cache[il,:])
            time_cache[il,:] *= wind
            
            #Fill the frequency cache
            freq_cache[il,:]= np.fft.rfft(time_cache[il,:])[1:]
            
        #Now fill the pre-stack cross-spectral density matrix
        for ii in range(np.size(freq_cache,1)):
            csdm[:,:,ii,i]= np.outer(freq_cache[:,ii],np.conj(freq_cache[:,ii])) * (1./(fc*we))
        
        #correction for single sided spectrum
        csdm[:,:,1:-1,i] *= 2 
            
        start += R
        stop = start+ nfft
        
        
    return csdm 
    
    
    
if __name__=="__main__":
    
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import ATS_Utils.ATSIO as atsio
    import matplotlib.gridspec as gds
    
    
    #REal DATA test
    path= r"C:\Test\TS_test\Toscana\PM-5\meas_2012-11-01_15-57-09\173_V01_C00_R000_TEx_BL_64H.ats"
    H= atsio.ReadATSHead(path)
    data= atsio.ReadATSData(path)* H['LSBVal'][0]
    fc= H['SRate'][0]
    N= 4000
    dt= 1./fc
    t= np.arange(0,N*dt,dt)
    x= data[10*N:11*N]
    
#    fc=120.
#    f1=2
#    f2=15
#    dt=1./fc
#    T= 30
#    N= np.int32(T/dt)
#    t= np.arange(0,N*dt,dt)
    
    #RECT test
#    NN=50
#    x= np.zeros_like(t)
#    mask= np.logical_and(t>=(N/2-NN)*dt,t<=(N/2+NN)*dt)
#    x[mask]= 1.0 * np.cos(2*np.pi*f2*t)
    
    # sinusoidal test
#    x= np.sin(2*np.pi*f1*t)
    
    #chirp test
#    k= 5
#    x= np.cos( 2*np.pi*f1*t+k*(t**2/2.) )
    
#    frist= ist_freq(x,fc,plot=False)
#    
#    #Plot ist.freq.
#    fig= plt.figure(figsize=(10,8))
#    fig.suptitle("Instantaneous frequency estimation for E channel - PST")
#    ax= fig.add_subplot(211)
#    ax.plot(t, frist,'k-'),ax.grid(True),
#    ax.set_ylabel("Inst. Frequency [Hz]"),
#    ax.set_xlabel("Time [s]"),
#    ax2= fig.add_subplot(212)
#    ax2.plot(t, x, 'b-'),ax2.grid(True)
#    ax2.set_ylabel("Amplitude"),
#    ax2.set_xlabel("Time [s]")
    
    #Calculate WVD
    kdt= 2
    wdf= WDF(x,fc, dt=kdt)
    freq= np.fft.fftfreq(2*N,dt)[:N+1]
    freq[-1] *= -1
    timeax= np.arange(0,N*dt,dt*kdt)
    
    fig= plt.figure(figsize=(16,12))
    
    GD1= gds.GridSpec(3,7)
    gs1= GD1.new_subplotspec( [0,1], 3, 6)
    gs2= GD1.new_subplotspec( [0,0], 3, 1)
    ax1= fig.add_subplot(gs1)
    ax2= fig.add_subplot(gs2)
    
    maps= ax1.pcolormesh( freq,timeax, np.abs(wdf) ,cmap='rainbow', norm=LogNorm()),
    ax1.set_title("Wigner-Ville distribution")
    ax1.set_xlabel("Frequency [Hz]"),
    ax1.set_ylabel("Time [s]"),
    plt.colorbar(maps[0],ax=ax1)
    
    
    ax2.plot(x,t,'b-'), ax2.grid(True)
    
    #Calculate Frac FT
    #alpha order
#    angle= -0.05*np.pi
#    a= (2./np.pi)*angle
#    
#    Fra= FFrFT(x,a)
#    
#    #Calculate WVD
#    wdf= WDF(Fra,fc)
#    freq= np.fft.fftfreq(4*N,dt)[:2*N]
#    timeax= np.arange(0,N*dt,dt)
#    
#    fig2= plt.figure(figsize=(16,12))
#    ax2= fig2.add_subplot(111)
#    ax2.pcolormesh( freq,timeax,wdf ,cmap='gray'),
#    ax2.set_title("Wigner-Ville distribution rotated - alpha= "+str(np.rad2deg(angle)))
#    ax2.set_xlabel("Frequency [Hz]"),
#    ax2.set_ylabel("Time [s]"),
    
    plt.show()
    
    
    
#    fig= plt.figure(figsize=(16,12))
#    fig.suptitle("Fractional Fourier Transform example")
#    ax= fig.add_subplot(211)
#    ax.plot(t,x,'b-'), ax.grid(True),
#    ax.set_ylim(0,1.5)
#    ax2= fig.add_subplot(212)
#    ax2.plot(t,np.real(Fra),'r-'), ax2.grid(True),
#    ax2.plot(t,np.imag(Fra),'g-'),
#    
#    ax.legend(["TimeDomain - real"])
#    ax2.legend([str(a)+"-domain, real",str(a)+"-domain, imag"])
#    
#    plt.show()
    
    
    
    