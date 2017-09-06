# -*- coding: utf-8 -*-


import numpy as np



def psdcore(x, ntaper, ndecimate=1, psddata=None):
    """    
    %  function psd = psdcore(x,  ntaper)
    %  Compute a spectral estimate of the power spectral density
    %  (psd) for the time series x using sine multitapers.
    %  Normalised to sampling interavl of 1.
    %  
    %  ntaper gives the number of tapers to be used at each frequency:
    %  if ntaper is a scalar, use same value at all freqs; if a
    %  vector, use ntaper(j) sine tapers at frequency j. 
    %  If series length is n, psd is found at 1 + n/2 evenly spaced freqs;
    %  if n is odd, x is truncted by 1.
    
    %  ndecimate: number of psds actually computed = (1+n/2)/ndecimate;
    %  these values are linearly interpolated into psd.
    """
    
    if psddata == None:
        n = len(x);
        n = n - np.mod(n,2)    # Force series to be even in length
        nhalf = n/2
        varx = np.var(x[0:n])
        ntap = np.ones(nhalf+1)*ntaper;  # Make a vector from scalar value
        # Remove mean; pad with zeros
        z = np.concatenate((x[0:n] - np.mean(x[0:n]), np.zeros(n)));
        # Take double-length fft
        fftz = np.fft.fft(z);    
    else:
        ntap = ntaper;
        fftz = psddata['fftz']
        n = psddata['n']
        nhalf = psddata['nhalf']
        varx = psddata['varx']
    
   
    #  Select frequencies for PSD evaluation
    if len(ntaper) > 1 and ndecimate > 1:
        nsum = np.cumsum(1.0/ntap)
        tmp = nhalf * (nsum-nsum[0])/(nsum[-1]-nsum[0])
        f = np.concatenate((np.round(np.interp(np.r_[0:nhalf+ndecimate:ndecimate], tmp,np.r_[0:nhalf+1])), [nhalf]))
        f = np.unique(np.concatenate(([0], f))) # Remove repeat frequencies in the list
    else:
        f = np.r_[0:nhalf+1]

    #  Calculate the psd by averaging over tapered estimates
    nfreq = len(f)
    psd = np.zeros(nfreq);
    
    # Loop over frequency
    for j in range(nfreq):
    
       m = f[j]
       tapers = ntap[m]
       # Sum over taper indexes; weighting tapers parabolically
       k = np.r_[1:tapers+1]
       w = (tapers**2 - (k-1)**2)*(1.5/(tapers*(tapers-0.25)*(tapers+1)))
       j1 = np.array(np.mod(2*m+2*n-k, 2*n), dtype=int)
       j2 = np.array(np.mod(2*m+k, 2*n), dtype=int)
       psd[j] = np.dot(w, np.abs(fftz[j1]-fftz[j2])**2)
    
    
    #  Interpolate if necessary to uniform freq sampling
    if len(ntaper) > 1 and ndecimate > 1:
        psd = np.interp(np.r_[0:nhalf+1], f,psd)
    
    #  Normalize by variance
    area = (np.sum(psd) - psd[0]/2. - psd[-1]/2.)/nhalf;  # 2*Trapezoid
    psd = (2.0*varx/area)*psd;
    
    return psd,psddata


##############################################################

def riedsid(psd, ntaper):
    """        
    %  function kopt = riedsid(psd, ntaper)
    %  Estimates optimal number of tapers at each frequency of
    %  given psd, based on Riedel-Sidorenko MSE recipe and other
    %  tweaks due to RLP.  
    %  ntaper is the vector of taper lengths in the previous iteration.
    %
    %  Initialize with ntaper=scalar
    """
    
    eps=1e-78;  #  A small number to protect against zeros
    
    nf = len(psd)
    ntap = ntaper
    if (len(ntaper)==1):
        ntap=ntaper*np.ones(nf)
    
    nspan = np.array([min(0.5*nf, 1.4*_nt) for _nt in ntap], dtype=int)
    
    #  Create log psd, and pad to handle begnning and end values
    nadd = 1 + np.max(nspan);
    Y = np.log(eps + np.concatenate((psd[nadd-1:0:-1], psd, psd[nf-2:nf-nadd-1:-1])))
    
    #  R = psd"/psd = Y" + (Y')^2 ; 2nd form preferred for consistent smoothing
      
    d2Y = np.zeros(nf)
    dY=d2Y
    
    for j in range(nf):
        
        j1 = int(j-nspan[j]+nadd) 
        j2 = int(j+nspan[j]+nadd)
    
        # Over an interval proportional to taper length, fit a least
        #  squares quadratic to Y to estimate smoothed 1st, 2nd derivs
        u = np.r_[j1:j2+1] - (j1+j2)/2.0;
        L = j2-j1+1
        uzero=(L**2.-1.)/12.0;
    
        dY[j] = np.dot(u,Y[j1-1:j2])*(12./(L*(L**2.0-1.0)))
        d2Y[j] = np.dot(u**2 - uzero,Y[j1-1:j2])*(360./(L*(L**2.0-1.0)*(L**2.0-4.0)))
   
    
    
    #  Riedel-Sidorenko recipe: kopt = (12*abs(psd ./ d2psd)).^0.4; but
    #  parabolic weighting in psdcore requires: (480)^0.2*abs(psd./d2psd)^0.4
    #
    #  Original form:  kopt = 3.428*abs(psd ./ d2psd).^0.4;
    
    kopt = np.round( 3.428 / np.abs(eps + d2Y + dY**2)**0.4 );
    
    #  Curb run-away growth of kopt due to zeros of psd''; limits
    #  slopes to be < 1 in magnitude, preventing greedy averaging:
    #  Scan forward and create new series where slopes <= 1
    state=0
    for j in range(1,nf):
        if (state == 0):
            slope=kopt[j]-kopt[j-1]
            if slope >= 1:
                state=1
                kopt[j]=kopt[j-1]+1
        else:
            if kopt[j] >= kopt[j-1]+1:
                kopt[j] = kopt[j-1]+1
            else:
                state=0
   
    #  Scan backward to bound slopes >= -1
    state=0;
    for j in range(nf-1,0,-1):
        if (state == 0):
            slope=kopt[j-1]-kopt[j]
            if slope >= 1:
                state=1
                kopt[j-1]=kopt[j]+1
         
        else:
             if kopt[j-1] >= kopt[j]+1:
                 kopt[j-1] = kopt[j]+1
             else:
                 state=0
    
    #  Never average over more than the psd length!
    kopt = np.array([min(k, nf*np.round(nf/3.)) for k in kopt])
    
    return kopt



def psd(x, fsamp=1.0, **kwargs):
    """
    function [psd f] = pspectrum(x, fsamp)
    %  function [psd f] = pspectrum(x, fsamp)
    %  Adaptive multitaper estimator of power spectral density (psd) of
    %  the stationary time series x .
    %  fsamp is the sampling frequency =1/(sampling interval).  If fsamp
    %  is absent, use fsamp=1.
    %
    %  psd of length nf gives spectrum at nf evenly spaced frequencies: 
    %  f=[ 0, df, 2*df, ... (nf-1)*df]', where nf = 1+ n/2, and n=length(x),
    %  and df=1/T.  If n is odd, x is truncated by 1.
    %
    """
    # -------  Tuning parameters -------------
    #   Cap=maximum number of tapers allowed per freq; then uncertainty
    #   of estimates >= psd/sqrt(Cap).
    
    Cap = kwargs.get('Cap', 1000)
    
    #  Niter=number of refinement iterations; usually <= 5
    Niter = kwargs.get('Niter', 5);
    
    #   Ndecimate: number of actual psd calculations is n/Ndecimate;
    #   the rest are filled in with interpolation.  Sampling in
    #   frequency is variable to accommodate spectral shape
    Ndecimate = kwargs.get('Ndecimate',10)
    if len(x) < 10000:
        Ndecimate=1
    
    #  Get pilot estimate of psd with fixed number of tapers
    initap=kwargs.get('initap',20)
    
    psd,psddata = psdcore(x, [initap], 1, None)
    nf = len(psd);
    
    #  Iterative refinement of spectrum 
    ntaper = initap*np.ones(nf)
    for iterate in range(Niter):
    
      kopt = riedsid(psd, ntaper)
      ntaper = np.array([min(k,Cap) for k in kopt])

      psd,psddata = psdcore(x, ntaper, Ndecimate, psddata)
    
    #  Scale to physical units and provide frequency vector
    psd = psd/fsamp;
    f = np.linspace(0, fsamp/2.0, nf);
    
    return (f, psd)
    

if __name__=='__main__':
    import ATS_Utils.ATSIO as ATSIO
    import matplotlib.pyplot as plt
    
    head = ATSIO.ReadATSHead(r"H:\_SHARED_TEMP\to_calandrinir\from_SG\CopperCliff\305a\meas_2012-07-15_22-25-46\205_V01_C00_R000_TEx_BL_64H.ats")
    data = ATSIO.ReadATSData(r"H:\_SHARED_TEMP\to_calandrinir\from_SG\CopperCliff\305a\meas_2012-07-15_22-25-46\205_V01_C00_R000_TEx_BL_64H.ats")*head['LSBVal']
    (f,pxx)=psd(data[0:30000])
    
    fig= plt.figure(figsize=(12,12))
    ax= fig.add_subplot(111)
    ax.loglog(f,pxx, 'k'), ax.grid(True)
    
    plt.show()
    
