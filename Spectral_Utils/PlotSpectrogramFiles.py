'''
Created on Apr 18, 2012

@author: Dario Errigo e-mail: d.errigo@fugro.com
'''

import numpy as np
import sys,os, fnmatch
import ATS_Utils.ATSIO as ATSIO
import Tkinter as tk
import tkFileDialog
import argparse
import tempfile
import Image
import datetime, pytz
from matplotlib.dates import date2num, AutoDateFormatter, AutoDateLocator

#from matplotlib.backends.backend_pdf import PdfPages    
from matplotlib import pyplot as plt

root = tk.Tk()
root.withdraw()


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                 description = 'PlotSpectrogramFiles -- Plots spectrogram from ATS to image or to screen',
                                 prog = 'PlotSpectrogramFiles')
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-f', dest='files', metavar='filename', type=str, nargs='+',
                   help='Files to plot (ats)')
group.add_argument('-r', dest='rootdir', metavar='root_directory', type=str,
                   help='Root directory, all ATS files will be plotted after grouping them by directory')
group.add_argument('-i', '--interactive', action='store_true', help='Interactive mode')
parser.add_argument('-o', dest='output_directory', metavar='output_directory', type=str, help='Output directory')
parser.add_argument('-freq', dest='freq', type = float, nargs = 2, default = [], help = 'Lower and upper bound of frequency axes (ex: [min, max])')
parser.add_argument('-wl', dest='window_length',type = int, help='Window length')
parser.add_argument('-ov', dest='overlap_samples',type = int, help='# of overlap samples')
parser.add_argument('-auto_clim', '--auto_clim', action='store_true', help='Activate automatic color limit setting')
parser.add_argument('-clim_e', dest='clim_e',type = int, nargs=2, default = [-5, 2], help = 'Lower and upper bound of colorscale of the electrical data expressed in logarithmic scale (ex: [min, max])')
parser.add_argument('-clim_h', dest='clim_h',type = int, nargs=2, default = [-5, 2], help = 'Lower and upper bound of colorscale of the magnetic data expressed in logarithmic scale (ex: [min, max])')


results = parser.parse_args()

interactive = results.interactive

window_length = results.window_length

overlap_samples = results.overlap_samples

freqs = results.freq

#Get color limits from default or user defined
auto_clim= results.auto_clim

if not auto_clim:
    clim_e = results.clim_e
    clim_h = results.clim_h
else:
    clim_e= None
    clim_h= None

if window_length == None:
    window_length = 1024
    
if overlap_samples == None:
    # Fix over=0.5
    overlap_samples = window_length/2

if interactive:
    files = tkFileDialog.askopenfilenames(title='Select file(s) to plot',
                                          filetypes=['ATS {.ats}'])
    files = root.tk.splitlist(files)
    
elif results.files:
    files = results.files
    
elif results.rootdir:
    rootdir = results.rootdir
    files = []
    for base, dirs, sub_files in os.walk(rootdir):
        goodfiles = fnmatch.filter(sub_files, '*.ats')
        files.extend(os.path.join(base, f) for f in goodfiles)

if len(files)==0: 
    sys.exit(1)
if not results.output_directory:
    results.output_directory= os.path.join( os.path.dirname(files[0]),'Spectrograms' )
    if not os.path.exists(results.output_directory): 
        os.makedirs(results.output_directory)
else:      
    if not os.path.exists(results.output_directory): 
        os.makedirs(results.output_directory)

# Loop on selected files
for f in files:
    
    print 'Computing spectrogram for ' + os.path.basename(f)
    
    # Read some header infos
    H = ATSIO.ReadATSHead(f)
    starttime = H['StartSecond'][0]
    srate = H['SRate'][0]
    nsamp = H['NSamp'][0]
    title = H['ChType'][0]
    hlen = H['HLength'][0] # in bytes
    endtime = starttime + nsamp/srate
    
    #Open file in direct binary mode
    fATS = open(f, 'rb')
    
    # Memmap the file using the numpy facility [this limits the possible size of the file to 2GB]
    trace = np.memmap(fATS, dtype = np.int32, mode = 'r', offset = hlen)*H['LSBVal'][0]
    
    #Number of frequency bins for the semispectrum, assuming (window_length) is even
    freq_bin = window_length/2
    
    step = window_length - overlap_samples
    
    #Time window axis
    ind = np.arange(0, nsamp - window_length + 1, step)
    
    #Number of time window
    time_bin= len(ind)
    
    #Create a temp file
    spec = tempfile.TemporaryFile()

    # memory allocation for the spectrogram --> memmap write!
    spectro = np.memmap(spec, dtype = 'float32', mode = 'w+', shape = (freq_bin, time_bin))
    
    #set window type
    w = np.hanning(window_length)     

    # spectrogram computation
    t = []
    start = 0
    i = 0
    climax= 0.0
    climin= 100.0
    if nsamp < window_length*2:
        print 'Window length is too big!'
        continue 
    
    while start < nsamp - window_length + 1:
        
        #Segment preprocessing
        x = trace[start:start + window_length].copy()
        x -= np.mean(x)
        windowed_data = x * w
        window_norm2 = np.sum(w ** 2)
        
        #FFT
        fft_data = np.fft.rfft(windowed_data, window_length)
        
        #RAW spectrogram calculation
        abs_data = np.abs(fft_data[:freq_bin])**2 / (srate*window_norm2)
        abs_data = np.flipud(abs_data)
        if (window_length % 2 == 0):
            abs_data[1:freq_bin - 1] *= 2
        else:
            abs_data[1:freq_bin] *= 2
        abs_data = np.log10(abs_data)
        if np.max(abs_data)>climax:
            climax= np.max(abs_data)
        if np.min(abs_data)<climin:
            climin= np.min(abs_data) 
        spectro[:,i] = abs_data
        start = start + step
        i = i + 1
    
    print 'Done!\n'
    del trace, x, windowed_data, abs_data, fft_data 
    fATS.close()
    
    # image creation
    print 'Drawing the png file'
    time = np.linspace(starttime, endtime, time_bin)
    if freqs == []:
        faxis = np.linspace((srate/window_length),srate/2, 512)
    else:
        if freqs[1] > srate/2:
            freqs[1] = srate/2
        faxis = np.linspace(freqs[0],freqs[1], 512)
    
    taxis = np.linspace(time[0],time[-1], 1024)
    
    #Creating plot
    fig = plt.figure(figsize = (20,14))
    plt.axes([0,0.2,1,0.5])    
    
    #Set color limits
    if title == 'Ex' or title == 'Ey':
        if auto_clim:
            clim = [climin,climax]
        else:
            clim = clim_e
    else:
        if auto_clim:
            clim = [climin, climax]
        else:
            clim = clim_h
    
    #Set frequency limits
    flim_min = round(faxis[0]*window_length/srate)
    flim_max = round(faxis[-1]*window_length/srate)
    
    #Set range limits
    rge = clim[1] - clim[0]
    
    #Limiting the color space of the image when it's still in array format: 8 bit per pixel
    im = Image.fromarray(np.array(np.floor(255*(np.clip(np.flipud(spectro[flim_min:flim_max][:]),clim[0], clim[1]) - clim[0])/rge), dtype='uint8'))
    #reshape
    im = im.resize((time_bin,freq_bin))
    #reshape and convert back to float32
    im = np.array(np.reshape(im.getdata(), (freq_bin, time_bin)), dtype='float32')
    
    #Locator and date formatter
    xlocator = AutoDateLocator(minticks=3, maxticks=8, tz=pytz.utc)
    dateFormatter = AutoDateFormatter(xlocator, tz=pytz.utc)
    
    #Adjust color limits
    im = (im/255.)*rge + clim[0]
    date_taxis = date2num([datetime.datetime.fromtimestamp(float(t), tz = pytz.utc) for t in taxis])
    
    #plot as a colormesh
    plt.pcolormesh(date_taxis, faxis, im, vmin = clim[0], vmax = clim[1])

    plt.colorbar()
    plt.gca().set_yscale('log')
    plt.xlabel('time')
    plt.ylabel('freq (Hz)')
    plt.title(title)
    ax = plt.gca()
    ax.xaxis.set_major_locator(xlocator)
    ax.xaxis.set_major_formatter(dateFormatter)
#    xlocator.autoscale()
    ax.set_xlim(date_taxis[0], date_taxis[-1])
    ax.set_ylim(faxis[0] , faxis[-1])


    fig.savefig(os.path.join(results.output_directory,'Spectrogram_'+os.path.basename(f)+'.png'), format = 'png', bbox_inches = 'tight', pad_inches = 1)
    
    print 'Done!\n'
    del fig
    del spectro
    

print 'Operation completed \n'




