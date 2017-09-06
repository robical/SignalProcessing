#!/usr/bin/env python

"""
Filter data with a FIR filter using the overlap-add method.
"""

import numpy as np
from ATS_Utils import ATSIO

def nextpow2(x):
    """Return the first integer N such that 2**N >= abs(x)"""
    
    return np.ceil(np.log2(abs(x)))

def fftfilt(b, x, *n, **kwargs):
    """Filter the signal x with the FIR filter described by the
    coefficients in b using the overlap-add method. If the FFT
    length n is not specified, it and the overlap-add block length
    are selected so as to minimize the computational cost of
    the filtering operation."""
    
    N_x = len(x)
    N_b = len(b)

    # Determine the FFT length to use:
    if len(n):

        # Use the specified FFT length (rounded up to the nearest
        # power of 2), provided that it is no less than the filter
        # length:
        n = n[0]
        if n != int(n) or n <= 0:
            raise ValueError('n must be a nonnegative integer')
        if n < N_b:
            n = N_b
        N_fft = 2**nextpow2(n)
    else:

        if N_x > N_b:

            # When the filter length is smaller than the signal,
            # choose the FFT length and block size that minimize the
            # FLOPS cost. Since the cost for a length-N FFT is
            # (N/2)*log2(N) and the filtering operation of each block
            # involves 2 FFT operations and N multiplications, the
            # cost of the overlap-add method for 1 length-N block is
            # N*(1+log2(N)). For the sake of efficiency, only FFT
            # lengths that are powers of 2 are considered:
            N = 2**np.arange(np.ceil(np.log2(N_b)),np.floor(np.log2(N_x)))
            cost = np.ceil(N_x/(N-N_b+1))*N*(np.log2(N)+1)
            N_fft = N[np.argmin(cost)]

        else:

            # When the filter length is at least as long as the signal,
            # filter the signal using a single block:
            N_fft = 2**nextpow2(N_b+N_x-1)

    N_fft = int(N_fft)
    
    # Compute the block length:
    L = int(N_fft - N_b + 1)
    
    # Compute the transform of the filter:
    H = np.fft.fft(b,N_fft)

    decimate=kwargs.get('decimate')
    if decimate == True:
        y = np.zeros(N_x/2,float)
        stride=2
    else:
        y = np.zeros(N_x,float)        
        stride=1
        
    i = 0; iout = 0;
    while i <= N_x:
        il = min([i+L,N_x])
        k = min([i+N_fft,N_x])
        yt = np.fft.ifft(np.fft.fft(x[i:il],N_fft)*H,N_fft) # Overlap..
        chunk = yt[:k-i:stride]
        y[iout:iout+len(chunk)] += np.real(chunk) # and add
        i += L; iout += L/stride
    return y


def filtfilt_lm(b, inp, out, N=4096, **kwargs):
    """
    Block FIR filtering implementation with zero-phase (FWD-BWD) filter
    
    Freq. domain: Mixed OLS (first pass)-OLA (second pass) approach
    Time domain: classic fwd/bwd block filtering
    
    INPUTS:
        - b: filter
        - inp: array_like structure that support chunked indexing
        - out: array_like structure that support chunked indexing
        - t: time axis array
        - N: block size
        - test_mode: boolean, activate/deactivate test against np.filtfilt
        - domain_mode: 'T'/'F', impose the convolution method
        
    OUTPUTS:
        - None -> write directly to out

    """
    
    optparams={
               'test_mode': False,
               'domain_mode': 'T',
               'timeit': False,
               }
    
    optparams.update(kwargs)
    
    import scipy.signal as scisi
    
    b= np.asarray(b)       
    M= b.size       #filter length
    NN= inp.size    #input size
    
    if optparams['timeit']:
        import time
    
    if optparams['domain_mode']=='F':
        # Auxiliary variables initialization
        in_fft= np.zeros( N, dtype=np.complex128 )
        in2_fft= np.zeros( N, dtype=np.complex128 )
        out_mid= np.zeros( N-M+1, dtype=np.double )
        
        #Filter fft
        B= np.fft.rfft(b, n=N)
    
    if optparams['test_mode']:
        import matplotlib.pyplot as plt
        fig_tot= plt.figure(figsize=(12,12))
        ax1= fig_tot.add_subplot(111)
        ax1.grid(True)
        
        #Calculate with filtfilt for comparison
        try:
            filt2= scisi.filtfilt(b,[1],inp, padtype=None)[::2]
        except MemoryError:
            print "\nNumpy filtfilt can't process this TS, it's too long.."
    
    # Initialize counters
    start= 0
    stop= N
    start_out= 0
    stop_out= N/2.
    
    # Late stop is the total number of integer blocks that spans the entire
    # signal length
    late_stop= np.ceil(NN/N)*N
    if late_stop==NN:
        late_stop += N
        
    i=0
    if optparams['domain_mode']=='T':
        if optparams['timeit']:
            start_t= time.time()
        
        while( stop<=late_stop ):
                
            #first pass - time domain
            
            chunk= inp[start:stop]
            # lfilter already outputs a signal that has the same length of the
            # input
            zi= scisi.lfilter_zi(b, [1])
            itemp_conv, _= scisi.lfilter(b,[1],chunk, zi=zi*chunk[0])
            itemp_conv= itemp_conv[::-1]
                  
            #second pass - time domain
                
            temp_conv2, _= scisi.lfilter(b,[1], itemp_conv, \
                                         zi=zi*itemp_conv[0])
            filt_sig= (temp_conv2)[::-1]
            
            #Write decimated data to output buffer
            out[start_out:stop_out,0]= np.int32( np.round(filt_sig[::2].clip(-2**31, 2**31 - 1)) )
            
            start += N-M+1
            stop = start + N
            start_out += (N-M+1)/2.
            stop_out = start_out + N/2.
            i += 1
            
        if optparams['timeit']:
            elapsed= time.time()-start_t
            print "\nCalculation time: "+str(elapsed)+" [s]"
                
    elif optparams['domain_mode']=='F':
        if optparams['timeit']:
            start_t= time.time()
        
        while( stop<=late_stop ):
            ## Frequency domain method
    
            #Forward pass --> OLS
            in_fft= np.fft.rfft(inp[start:stop], n=N)
      
            # discard the first (M-1) samples --> OLS
            # these samples will be recovered in the second pass
            # so they won't be discarded here
            out_mid= np.real( np.fft.irfft(in_fft*B, n=N) )#[(M-1):]
              
            # Backward pass --> OLA
            # There's no overlap and add, just overlap; this way
            # one eliminates the M-1 samples that would be discarded 
            in2_fft= np.fft.rfft(out_mid, n=N)
              
            #IFFT, decimation, write to out
            try:
                temp= np.real( np.fft.irfft( in2_fft*np.conj(B), n=N ) )[::2]
                out[start_out:stop_out,0] = np.int32( np.round(temp.clip(-2**31, 2**31 - 1)) )
                
            except Exception:
                kN= out[start_out:stop_out,0].size
                temp= np.real( np.fft.irfft( in2_fft*np.conj(B), n=N ) )[::2]
                out[start_out:stop_out,0] = np.int32( np.round(temp[:kN].clip(-2**31, 2**31 - 1)) )
            
            start += N-M+1
            stop = start + N
            start_out += (N-M+1)/2.
            stop_out = start_out + N/2.
            i += 1
            
        if optparams['timeit']:
            elapsed= time.time()-start_t
            print "\nCalculation time: "+str(elapsed)+" [s]"
    else:
        #debug
        
        #Variables setup
        in_fft= np.zeros( N, dtype=np.complex128 )
        in2_fft= np.zeros( N, dtype=np.complex128 )
        out_mid= np.zeros( N-M+1, dtype=np.double )
        
        #Filter fft
        B= np.fft.rfft(b, n=N)
        
        while( stop<=late_stop ):
            
            chunk= inp[start:stop]
            
            #FILTFILT
            filt2= scisi.filtfilt(b, [1], chunk, padtype=None)[::2]
            
            #TIME DOMAIN
            
            # lfilter already outputs a signal that has the same length of the
            # input
            zi= scisi.lfilter_zi(b, [1])
            itemp_conv, _= scisi.lfilter(b,[1],chunk, zi=zi*chunk[0])
            itemp_conv= itemp_conv[::-1]
                  
            #second pass - time domain
            temp_conv2, _= scisi.lfilter(b,[1], itemp_conv, \
                                         zi=zi*itemp_conv[0])
            filt_sig= (temp_conv2)[::-1]
            
            #Write decimated data to output buffer
            out_time= filt_sig[::2]
            
            #FREQUENCY DOMAIN
            
            #Forward pass --> OLS
            in_fft= np.fft.rfft(chunk, n=N)
      
            #discard the first (M-1) samples --> OLS
            out_mid= np.real( np.fft.irfft(in_fft*B, n=N) ) #[(M-1):]
              
            # Backward pass --> OLA
            #out_mid= qc.zeropad(out_mid, (M-1))
            in2_fft= np.fft.rfft(out_mid, n=N)
              
            #IFFT, decimation, write to out
            out_freq = \
            np.real( np.fft.irfft( in2_fft*np.conj(B), n=N ) )
            out_freq[-(M-1)/2.:]= 0.
            out_freq= out_freq[::2]
            
            fig_comp= plt.figure(figsize=(10,10))
            ax= fig_comp.add_subplot(111)
            ax.stem( range(out_time.size), out_time, linefmt='b-', markerfmt='bo', basefmt='r-'),ax.grid(True)
            ax.stem( range(out_freq.size), out_freq, linefmt='g-', markerfmt='go', basefmt='r-')
            ax.stem( range(filt2.size), filt2, linefmt='m-', markerfmt='mo', basefmt='r-' )
            ax.legend(["TD filtering","FD filtering","FILTFILT (nopad)"])
            
            plt.show()
    

def OLA(sig, fil, mode='full', testplot=False, decimate=False):
    """
    Implements overlap-add method to filter long sequences in the frequency 
    domain.
    Useful in case of very long filter impulse response or signal.
    
    If the filter length is less than 20 samples, the convolution is done in 
    time domain.
    """
    
    if testplot:
        import matplotlib.pyplot as plt
    
    #Get signal and filter length
    kN= sig.size
    M= fil.size
    
    #The length of the signal pieces must be: power of two, greater than M, not
    # too large select next power of two greater than M, if M> Nmax, otherwise 
    # Nmax
    if(M<=16):
        #convolution in time domain
        result= np.convolve(sig, fil)
        
        #cut out the delay shift introduced by the filter
        if(np.mod(M,2)==0):
            return result[M/2:]
        else:
            return result[(M-1)/2:]
        
    else:
        #convolution in frequency domain with OLA
        Nfft= 2**( np.ceil(np.log2(2*M-1)) )
        MM= Nfft-M+1 #efficient win length
        R= MM #use rect window
        
        #Now for each pieces
        Nframes= 2+ np.floor( (kN-MM)/R ) #number of complete frames
        temp_s= np.zeros( Nfft ) #temporary var for signal piece
        temp_f= np.zeros( Nfft ) #temp for filter
        temp_f[:M]= fil #insert filter
        result= np.zeros( kN+M-1 ) #result of the acyclic convolution
        for i in range(int(Nframes)):
            #Load data in the input buffer
            index= np.arange(i*R,np.min( [i*R+MM,kN] ), dtype=np.int32)
            
            #insert signal piece
            temp_s[:index.size]= sig[index]
            
            #transform, multiplication and inverse transform
            outindex= np.arange(i*R,np.min([i*R+Nfft,kN+M-1]), dtype=np.int32)
            result[outindex] += \
            np.real( np.fft.irfft( np.fft.rfft(temp_s)*\
                                   np.fft.rfft(temp_f) ) )[:outindex.size]
            if testplot:
                fig= plt.figure(figsize=(16,12))
                ax= fig.add_subplot(111)
                ax.grid(True)
                ax.plot(np.arange(0,kN+M-1,1),result,'r-')
                plt.show()
            #clean input buffer
            temp_s= np.zeros( Nfft )
        
        if mode=='full':
            return result
        elif mode=='same':
            if(np.mod(M,2)==0):
                res= result[M/2:-(M/2-1):]
                if decimate:
                    return res[::2]
                else:
                    return res
            else:
                res= result[(M-1)/2:-(M-1)/2:]
                if decimate:
                    return res[::2]
                else:
                    return res
        else:
            print "Mode not recognized: a full convolution result will be "+\
            "returned."
            return result
        
def eval_IIRl(b,a, tol_time=1e-9,tol_freq=1e-5,maxiter=5,plot=False, 
              retfil=False):
    """
    This routine approximately evaluates an IIR filter impulse response 
    effective length.
    The number of coefficients is truncated when the abs() value, is below the 
    threshold (tol_time).
    
    The returned value is necessary for the OLAfiltfilt routine, in order to 
    evaluate the correct segment
    length that doesn't incur in severe time aliasing
    
    INPUT:
    b= numerator coeff. of the filter
    a= denominator coeff. of the filter
    tol= tolerance; every coeff. below this threshold is truncated
    
    OUTPUT:
    Lh= approx. impulse response length
    g= approx. impulse response (optional)
    """
    
    if plot:
        import matplotlib.pyplot as plt
    
    #initialization
    itera= 0
    Nfft= 128.
    
    #first calculation
    
    #evaluates filter frequency response on the unit circle
    w= np.arange( 0,np.pi+2*(np.pi/float(Nfft)),2*(np.pi/float(Nfft)) )
    B= np.zeros_like(w, dtype=np.complex128)
    A= np.zeros_like(w, dtype=np.complex128)
    for indkk,kk in enumerate(b):
        B += kk*np.exp(-1j*w*indkk)
        
    for indgg,gg in enumerate(a):
        A += gg*np.exp(-1j*w*indgg)
        
    G= B/A
    
    if plot:
        fig= plt.figure(1,figsize=(12,12))
        ax= fig.add_subplot(211)
        ax.semilogy( w,np.abs(G), 'r-' ), ax.grid(True),
        ax.set_xlabel('Normalized w'),ax.set_ylabel('Amplitude')
        
        ax2= fig.add_subplot(212)
        ax2.plot( w, np.rad2deg( np.unwrap(np.angle(G)) ), 'k-' ) 
        ax2.grid(True)
        ax2.set_xlabel('Normalized w'),ax2.set_ylabel('Phase')
    
    #calculates impulse response
    g= np.real( np.fft.irfft(G,int(Nfft) ) )
    
    if plot:
        fig2= plt.figure(2,figsize=(12,12))
        ax3= fig2.add_subplot(111)
        ax3.plot( g, 'b-' ), ax3.grid(True)
    
    # Truncation of the filter
    # gap usefulness: doesn't check the first samples, because there's the 
    # first rise of the impulse response
    trunked= False
    gap=10
    integr= np.cumsum( np.abs(g) )
    dd= np.diff(integr[gap:]) 
    dd_mask= dd<=tol_time
    for indi,i in enumerate(dd_mask):
        if i:
            g= g[:(indi-1)+gap]
            trunked= True
            break
        
    # Now it would be possible to check again for consistency btw
    # the frequency response of the complete IRR and the one generated
    # by the truncated IRR; check if it respects the threshold in frequency 
    # domain
    Gtrunc= np.fft.rfft( g, Nfft )
    
    # other cycles, if necessary
    
    while( (not(trunked) or np.sum(np.abs(np.abs(G)-np.abs(Gtrunc)))> \
            tol_freq) and itera<=maxiter ):
        #iteration counter
        itera += 1
        
        #next power of 2
        Nfft= 2**(np.log2(Nfft)+1)
        
        #evaluates filter frequency response on the unit circle
        w= np.arange(0,np.pi+2*(np.pi/float(Nfft)),2*(np.pi/float(Nfft)) )
        B= np.zeros_like(w, dtype=np.complex128)
        A= np.zeros_like(w, dtype=np.complex128)
        for indkk,kk in enumerate(b):
            B += kk*np.exp(-1j*w*indkk)
            
        for indgg,gg in enumerate(a):
            A += gg*np.exp(-1j*w*indgg)
            
        G= B/A
        
        if plot:
            fig= plt.figure(1,figsize=(12,12))
            ax= fig.add_subplot(211)
            ax.semilogy( w,np.abs(G), 'r-' ), ax.grid(True),
            ax.set_xlabel('Normalized w'),ax.set_ylabel('Amplitude')
            
            ax2= fig.add_subplot(212)
            ax2.plot( w, np.rad2deg( np.unwrap(np.angle(G)) ), 'k-' ) 
            ax2.grid(True)
            ax2.set_xlabel('Normalized w'),ax2.set_ylabel('Phase')
            
        #calculates impulse response
        g= np.real( np.fft.irfft(G,int(Nfft) ) )
        
        if plot:
            fig2= plt.figure(2,figsize=(12,12))
            ax3= fig2.add_subplot(111)
            ax3.plot( g, 'b-' ), ax3.grid(True)
    
        # Truncation of the filter
        # gap usefulness: doesn't check the first samples, because there's the 
        # first rise of the impulse response
        gap=10
        integr= np.cumsum( np.abs(g) )
        dd= np.diff(integr[gap:]) 
        dd_mask= dd<=tol_time
        for indi,i in enumerate(dd_mask):
            if i:
                g= g[:(indi-1)+gap]
                trunked= True
                break
            
        # Now it would be possible to check again for consistency btw
        # the frequency response of the complete IRR and the one generated
        # by the truncated IRR; check if it respects the threshold in frequency
        # domain
        Gtrunc= np.fft.rfft( g, Nfft )
        
    
    if plot:
        plt.show()
            
    if retfil:
        if (itera-1)==maxiter:
            print "Maximum number of iterations has been reached!"
        return len(g),g
    else:
        if (itera-1)==maxiter:
            print "Maximum number of iterations has been reached!"
        return len(g)

    
        
        
def OLAfiltfilt_iir( b,a,sig ):
    """
    This routine implements the forward/backward filtering method
    using the overlap-add technique to filter a very long input 
    signal with an IIR filter with zero phase
    """
    
    #calculates the B and A coefficient of the FWD/BWD filter
    B= np.convolve(b, 1./b)
    A= np.convolve(a, 1./a)
    
    #Evaluates a truncated IIR impulse response for the filter
    _,g= eval_IIRl(B,A, tol_time=1e-9,tol_freq=1e-5,maxiter=5,plot=True, \
                    retfil=True)
    
    #Use it in the OLA routine
    res= OLA(sig, g, mode='same', testplot=True, decimate=True)
    
    return res
    
    
        
        
def OLS(sig,fil, mode='full'):
    """
    Implements overlap-save method to filter long sequences in frequency domain
    
    Usually more memory efficient than OLA
    
    --> (TO BE COMPLETED, still little mismatch with TD and FD OLA results) <--
    """
    #Get signal and filter length
    kN= sig.size
    M= fil.size
    
    if(kN<=10*M):
        print "The length of the signal doesn't justify the use of OLS "+\
        "method: it's better to do time domain convolution."
        result= np.convolve(sig,fil,mode=mode)
        
        return result
    else:
        #Choose good chunking size
        #greater than M but not too large to slow down FFT computation
        MM= np.min( [10*M,8192] )
        Nfft= 2**(np.ceil(np.log2(MM))) #input chunk size
        rid= M-1 # samples to get rid of
        R= Nfft-rid #hop size
        Nframe= 2+ np.floor( (kN-R)/R)
        sig= np.concatenate( (sig, np.zeros(Nframe*Nfft-kN)) )
        temp_s= np.zeros(Nfft)
        result= np.zeros( Nframe*Nfft, dtype=np.double )
        for i in range(int(Nframe)):
            index= np.arange(i*R,i*R+Nfft, dtype=np.int32)
            temp_s= sig[index]
            outindex= np.arange(i*R,i*R+R, dtype=np.int32)
            result[outindex]= \
            np.real( np.fft.irfft( np.fft.rfft(temp_s)*\
                                   np.fft.rfft(fil, n=Nfft) ) )[rid:]
            
        if mode=='full':
            return result[:kN+(M-1)/2]
        elif mode=='same':
            if(np.mod(M,2)==0):
                return result[M/2:kN]
            else:
                return result[(M-1)/2:kN+(M-1)/2]
        else:
            print "Mode not recognized: a full convolution result will be"+\
            " returned."
            return result
            

def interp1D_f(data, fat):
    """
    1D exact interpolation on a regular grid using FFT.
    
    INPUTS:
        fat= interpolation factor (ex. 2 means the sampling rate of the input 
        signal will be doubled)
        sig= input signal
    OUTPUTS:
        xint= interpolated signal, #samples= fat*(sig.size)
        
    """
    sig= data.copy()
    
    #Remove DC
    m_sig= np.mean(sig)
    sig -= m_sig
    
    #Pad zeros to power of 2
    N= sig.size
    Nfft= np.int32( 2**(np.ceil( np.log2(N) ) ) )
    
    #Create interpolated frequency axis and calculates FFT(Nfft)
    Sfat= np.zeros( fat*Nfft , dtype= np.complex128)
    
    #Transform
    S= np.fft.fft(sig,n=Nfft)
    
    #Fill the interpolated spectrum
    Sfat[:Nfft/2]= S[:Nfft/2]
    
    #Share the Nyquist
    Sfat[Nfft/2+1]= S[Nfft/2+1]/2.
    Sfat[fat*Nfft- (Nfft/2+1)]= S[Nfft/2+1]/2.
    
    #Assign the negative frequencies part
    Sfat[-Nfft/2+2:]= S[-Nfft/2+2:]
    
    #Shift, apply window in frequency domain, deshift
#    Sfat= np.fft.fftshift(Sfat)
#    Sfat = Sfat* np.hamming( (fat*Nfft) )
#    Sfat= np.fft.ifftshift(Sfat)
    
    #Return back in time domain
    xint= np.real( np.fft.ifft(Sfat) )*fat +m_sig
    
    
    return xint[:fat*N]

def FIRdecimator(cut_off, trans_width, att_stop, fc):
    """
    Returns a decimation filter calculated with the parameters
    given in input using the window method, with a Kaiser window
    """
    
    import scipy.signal as scisi
    
    # The Nyquist rate of the signal.
    nyq_rate = fc / 2.0
 
    # The desired width of the transition from pass to stop,
    # relative to the Nyquist rate.  We'll design the filter
    # with a 5 Hz transition width.
    width = trans_width/nyq_rate
 
    # The desired attenuation in the stop band, in dB.
    ripple_db = att_stop

    # Compute the order and Kaiser parameter for the FIR filter.
    N, beta = scisi.kaiserord(ripple_db, width)
 
 
    # Use firwin with a Kaiser window to create a lowpass FIR filter.
    b = scisi.firwin(N, cut_off/nyq_rate, window=('kaiser', beta))
    
    return b


if __name__=="__main__":
    
    import ATS_Utils.ATSIO as atsio
    import scipy.signal as scisi
    import Spectral_Utils.QCutils as qc
    import matplotlib.pyplot as plt
    import os
        
    #synthetic data: test on pure sinusoid
    fc= 300.
    dt= 1./fc
    NN= 1000000
    t= np.arange( 0,NN*dt,dt )
    inp= np.sin(2*np.pi*25*t )
    
    #Set up decimator
    cut_off= 0.45*(fc/2.)
    att_stop= 60.0
    trans_width= 25.
    b= FIRdecimator(cut_off, trans_width, att_stop, fc)
    
    #Load input and create time axis; reserve space for output
    #t= np.arange(inp.size)/fc
    out= np.zeros( (inp.size/2.,1), dtype= np.double )
    
    filtfilt_lm(b, inp, out, N=32768, test_mode=False, domain_mode='F',\
                timeit=True)
                
                
    
    
    #Test phase preservation
    M= 32768
    over= 0.0
    wind= np.hanning(M)
    fk_in,phase_in= qc.phase_est(inp,M,over,wind,fc, unwrap=False)
    fk_out,phase_out= qc.phase_est(out,M,over,wind,fc/2., unwrap=False)
    
    #Comparison with original phase
    phase_orig= np.unwrap( np.angle( np.fft.rfft(inp)[1:] ) )
    fk_orig= np.fft.fftfreq(inp.size, 1./fc)[1:inp.size/2+1]
    fk_orig[-1] *= -1
    
    fig= plt.figure( figsize=(10,10) )
    ax= fig.add_subplot(111)
    ax.plot( fk_in, np.rad2deg( np.unwrap( phase_in ) ),'k-' ), ax.grid(True)
    ax.plot( fk_out, np.rad2deg( np.unwrap( phase_out ) ), 'r-' ),
    ax.plot( fk_orig, np.rad2deg( phase_orig ), 'g-' )
    ax.legend(["Original phase [deg]","Decimated phase [deg]","Original Phase [deg] - (noBlock)"])
    
    plt.show()
    
    #Convert the decimated TS to ATS format and write out
#     h_out= head.copy()
#     h_out['SRate']= float(fc/2.)
#     h_out['LSBVal']= ( np.max(out)-np.min(out) )/(2**32)
#     data_filt= out/h_out['LSBVal']
#     data_filt= np.int32(data_filt.clip(-2**31, 2**31 - 1))
#     atsio.WriteATSData(out_name, h_out, data_filt)
#     
#     print "\nThe decimated TS "+os.path.basename(out_name)+" has been written!"