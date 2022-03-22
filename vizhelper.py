import pprint
import numpy as np
import glob
import os, shutil
import scipy
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.figure_factory as ff

from casatasks import visstat
from casatools import table
from casatools import componentlist
from casatools import simulator
from casatasks.private import simutil
from prettytable import PrettyTable
from matplotlib import colors as mcolors
from plotly.subplots import make_subplots

cl = componentlist()
sm = simulator()

def make_ms_frame(msname:str, ant_config:str, spw_params=None, verbose=False):
    """ 
    Construct an empty MeasurementSet that has the desired observation setup. 
        
    Args:
        msname: MS to create
        ant_config: a telescope configuration from casadata (alma/simmos) (for 
                  example  alma/simmos/aca.cycle8.cfg)
        swp_params: parameters such as SPW name, number of channels, frequency
    """ 
            
    ## Open the simulatorFunctions
    sm.open(ms=msname);

    ## Read/create an antenna configuration. 
    ## Canned antenna config text files are located here: <casadata>/alma/simmos/*cfg
        
    antennalist = os.path.join(ctsys.resolve("alma/simmos"), ant_config)
    if verbose:
        print(f'Using antenna list file: {antennalist}')
            
        
    ## Fictitious telescopes can be simulated by specifying x, y, z, d, an, telname, antpos.
    ##     x,y,z are locations in meters in ITRF (Earth centered) coordinates. 
    ##     d, an are lists of antenna diameter and name.
    ##     telname and obspos are the name and coordinates of the observatory. 
        
    mysu = simutil.simutil()
    (x, y, z, d, an, an2, telname, obspos) = mysu.readantenna(antennalist)

    ## Set the antenna configuration
    metool = measures()
    sm.setconfig(
        telescopename=telname,
        x=x,
        y=y,
        z=z,
        dishdiameter=d,
        mount=['alt-az'], 
        antname=an,
        coordsystem='local',
        referencelocation=metool.observatory(telname)
    )

    ## Set the polarization mode (this goes to the FEED subtable)
    sm.setfeed(mode='perfect X Y', pol=['']);    # X Y / R L

    ## Set the spectral window and polarization (one data-description-id). 
    ## Call multiple times with different names for multiple SPWs or pol setups.
    sm.setspwindow(
        spwname=spw_params['name'],
        freq=spw_params['freq'],
        deltafreq='0.1GHz',
        freqresolution='0.2GHz',
        nchannels=spw_params['nchan'],
        stokes='XX YY'
    )

    ## Setup source/field information (i.e. where the observation phase center is)
    ## Call multiple times for different pointings or source locations.
    source_name = 'simulated_source'
    sm.setfield(
        sourcename=source_name,
        sourcedirection=metool.direction(
            rf='J2000',
            v0='19h59m28.5s',
            v1='+40d44m01.5s'
        )
    )

    ## Set shadow/elevation limits (if you care). These set flags.
    sm.setlimits(shadowlimit=0.01, elevationlimit='1deg');

    ## Leave autocorrelations out of the MS.
    sm.setauto(autocorrwt=0.0);  

    ## Set the integration time, and the convention to use for timerange specification
    ## Note : It is convenient to pick the hourangle mode as all times specified in sm.observe()
    ##        will be relative to when the source transits.
    sm.settimes(
        integrationtime='2s',
        usehourangle=True,
        referencetime=metool.epoch('UTC','2021/10/14/00:01:02')
    )

    ## Construct MS metadata and UVW values for one scan and did 
    ## Call multiple times for multiple scans.
    ## Call this with different sourcenames (fields) and spw/pol settings as defined above.
    ## Timesteps will be defined in intervals of 'integrationtime', between starttime and stoptime.
        
    sm.observe(
        sourcename=source_name,
        spwname=spw_params['name'],
        starttime='0', 
        stoptime='+8s'
    )

    ## Close the simulator
    sm.close()
      
    ## Unflag everything (unless you care about elevation/shadow flags)
    flagdata(vis=msname, mode='unflag', flagbackup=False)
    
    
def add_gaussian_noise(msname:str, noise_level_sigma='0.1Jy'):
    """
    Adds Gaussian random noise using simulator / corrupt
  
    Args:
        ms_name: MS to modify
        noise_level_sigma: noise sigma as used in the simulator corrupt function
    """
        
    try:
        sm.openfromms(msname)
        sm.setseed(4321)
        sm.setnoise(mode='simplenoise', simplenoise=noise_level_sigma)
        sm.corrupt()
    finally:
        sm.close()
            
    
def make_point_source_comp_list(cl_name:str, freq:str, flux:str, spectrumtype:str, index:int, 
                                direction='J2000 19h59m28.5s +40d44m01.5s',
                                fluxunit='Jy', shape='point', label='sim_point_source'):
    """
    Makes a component list file with a point source
  
    Args:
        cl_name: name of the component list file to create
        freq: freq quantity (with units) as used in the component list tool 
        flux: flux, units are assumed in Jy
        spectrumtype: as used in component list tool: constant/spectral index
        index: spectral index
    """
        
    try:
        cl.addcomponent(
            dir=direction, 
            flux=flux,
            fluxunit=fluxunit, 
            freq=freq,
            shape=shape,     # shape='gaussian',
                             # majoraxis="5.0arcmin", minoraxis='2.0arcmin',                                
                             # polarization='linear', / 'Stokes'
                             # spectrumtype:'spectral index' / 'constant'
            spectrumtype=spectrumtype,
            index=index,
            label=label)
        cl.rename(filename=cl_name)
    finally:
        cl.close()
        
def sim_from_comp_list(msname:str, cl_name:str):
    """
    Updates the MS visibilities using simulator.predict to add
    components from the components list

    Args:
        ms_name: MS to modify
        cl_name: name of components list file to simulate
    """
    try:
        sm.openfromms(msname)
        sm.predict(complist=cl_name, incremental=False)
    finally:
        sm.close()
            
def add_point_source_component(msname, cl_name = 'sim_point_source.cl', freq=None, 
                                flux=5.0, spectrumtype='constant', index=-1):
    """
    Adds a point source to the MS
 
    Args:
        ms_name: MS to modify
        freq: 
        spectrumtype:
        flux: In Jy, as used in componentlist.addcomponent
    """
        
    make_point_source_comp_list(
        cl_name=cl_name, 
        freq=freq, 
        flux=flux,  
        spectrumtype=spectrumtype, 
        index=index
    )
        
    sim_from_comp_list(msname, cl_name)
    shutil.rmtree(cl_name)
        
def add_spectral_line(msname:str, line:'numpy.array', chan_range=[60, 86], amp_factor=None):
    """
    Adds a spectral line as a Gaussian function in the range of channels given
  
    Args:
        ms_name: MS to modify
        line: function to produce spectral line. ex: fn(x, mu, sigma)
        chan_range: list with indices of the first and last channel
        amp_factor: factor to multiply the peak height / flux density
    """
        
    try:
        tbtool = table()
        tbtool.open(msname, nomodify=False)
        data = tbtool.getcol('DATA')

            
            
        if not amp_factor:
            amp_factor = 1.0/np.max(line)
            
        data[:,chan_range[0]:chan_range[1],:] += amp_factor * (1+0j) * line.reshape((1, len(line), 1))
        tbtool.putcol('DATA', data)
        
    finally:
        tbtool.done()
            
def add_polynomial_continuum(msname:str, pol_coeffs:list, nchan:int, amp_factor=1.0):
    """
    Update MS visibilities adding a polynomial evaluated for all channels.
        
    Args:
        ms_name: MS to modify
        pol_coeff: polynomial coefficients, evaluated [0.5, 0.25] => 0.5x + 0.25
        nchan: number of channels in the SPW (x axis to eval polynomial)
    """
    try:
        tbtool = table()
        tbtool.open(msname, nomodify=False)
        data = tbtool.getcol('DATA')

        x = np.linspace(0, 1, nchan)
        polynomial = np.polyval(pol_coeffs, x)
            
        # Add same polynomial to real and imag part
        data += amp_factor * (1+1j) * polynomial.reshape((1, len(polynomial),1))

        tbtool.putcol('DATA', data)
    finally:
        tbtool.done()
        
def plot_ms_data(msname='sim_data.ms', myplot='uv', fitline=None, average=False)->None:
    """
    Plot all channels and samples of msdata file (amplitude vs channel). Samples indicated 
    as different colors.

    Options : myplot='uv'
              myplot='data_spectrum'
    Args:
      plot_complex: 'abs', 'real', or 'imag'
    """
    
    tb = table()
    from matplotlib.collections import LineCollection
    tb.open(msname)

    # UV coverage plot
    if myplot=='uv':
        plt.figure(figsize=(4,4))
        plt.clf()
        
        uvw = tb.getcol('UVW')
        
        plt.plot( uvw[0], uvw[1], '.')
        plt.plot( -uvw[0], -uvw[1], '.')
        plt.title('UV Coverage')
    
    # Spectrum of chosen column. Make a linecollection out of each row in the MS.
    spectrum = ['data_spectrum', 'corr_spectrum', 'resdata_spectrum', 'model_spectrum']
    
    if myplot in spectrum:
        dats=None
        if myplot=='data_spectrum':
            dats = tb.getcol('DATA')
        if myplot=='corr_spectrum':
            dats = tb.getcol('CORRECTED_DATA')
        if myplot=='resdata_spectrum':
            dats = tb.getcol('DATA') - tb.getcol('MODEL_DATA') 
        if myplot=='rescorr_spectrum':
            dats = tb.getcol('CORRECTED_DATA') - tb.getcol('MODEL_DATA') 
        if myplot=='model_spectrum':
            dats = tb.getcol('MODEL_DATA')
            
        xs = np.zeros((dats.shape[2],dats.shape[1]),'int')
        for chan in range(0,dats.shape[1]):
            xs[:,chan] = chan
    
        npl = dats.shape[0]
        fig, ax = plt.subplots(npl, 1, figsize=(30, 20))

        if average:
            for pol in range(0,dats.shape[0]):
                x = xs
                y = np.mean(dats[pol,:,:].real.T, axis=0)
                
                y_std = np.std(dats[pol,:,:].real.T, axis=0)
            
                ax[pol].scatter(x=xs[0], y=y, c='#ff8d33')
                ax[pol].plot(xs[0], y, c='#ff8d33')
                
                if fitline is not None:
                    y_fit = np.mean(fitline[pol,:,:].real.T, axis=0)
                    ax[pol].plot(xs[0], y_fit, color='black', linewidth=3)
                
                ax[pol].fill_between(
                    x=xs[0], 
                    y1=y + y_std,
                    y2=y - y_std,
                    alpha=0.2,
                    edgecolor='#ff8d33',
                    facecolor='#f0b27a'
                )
                ax[pol].set_title(myplot + ': polar: '+ str(pol) + " (real)")
                ax[pol].set_xlim(x.min(), x.max())
                ax[pol].set_ylim(dats[pol,:,:].real.T.min(), dats[pol,:,:].real.T.max())
            
        else:
            colors = [
                mcolors.to_rgba(c) for c in plt.rcParams['axes.prop_cycle'].by_key()['color']
            ]
        
            for pol in range(0,dats.shape[0]):
                x = xs
                y = dats[pol,:,:].real.T 
            
                data_real = np.stack((x, y), axis=2)
            
                ax[pol].add_collection(LineCollection(data_real, colors=colors))
                ax[pol].set_title(myplot + ': polar: '+ str(pol) + " (real)")
                ax[pol].set_xlim(x.min(), x.max())
                ax[pol].set_ylim(y.min(), y.max())
        
        plt.show()

def plot_ms_corrected_data(msname:str, myplot:str, chan_list:list)->None:
    """
    Plot corrected ms data as a amplitude vs channel, averaged along the sample axis. Errors bars are 
    shown per point as average stddev along the sample axis.

    Options : myplot='uv'
              myplot='data_spectrum'
    Args:
      plot_complex: 'abs', 'real', or 'imag'
    """
    
    baseline, base_chans = get_continuum_data(
        msname=msname, 
        myplot='data_spectrum', 
        chan_list=[chan_list]
    )
    
    peak, peak_chans = get_peak_data(
        msname=msname,
        myplot='data_spectrum', 
        chan_list=[chan_list]
    )
    
    npl = baseline.shape[0]
    
    fig, ax = plt.subplots(npl, 1, figsize=(30, 20))
    
    
    for pol in range(0, baseline.shape[0]):
        
        x = base_chans
        y = np.mean(baseline[pol,:,:].real.T, axis=0)
        
        xp = peak_chans
        yp = np.mean(peak[pol,:,:].real.T, axis=0)
                
        y_std = np.std(baseline[pol,:,:].real.T, axis=0)
        yp_std = np.std(peak[pol,:,:].real.T, axis=0)
            
        ax[pol].scatter(x=x, y=y, c='#ff8d33', marker='o')
        ax[pol].errorbar(x=x, y=y, yerr=y_std, c='#ff8d33')
        
        ax[pol].scatter(x=xp, y=yp, c='#3368ff', marker='o')
        ax[pol].errorbar(x=xp, y=yp, yerr=yp_std, c='#3368ff') 
        
        ax[pol].set_title(myplot + ': polar: '+ str(pol) + " (real)")
        ax[pol].set_xlim(x.min(), x.max())
        ax[pol].set_ylim(baseline[pol,:,:].real.T.min(), peak[pol,:,:].real.T.max())
    

        
def get_spectrum_data(msname:str, myplot:str)->'numpy.array':
    """
    Return spectrum data as numpy array

              myplot='data_spectrum'
      Args:
          plot_complex: 'abs', 'real', or 'imag'
    """
    from casatools import table
    import numpy as np

    tb = table()
    from matplotlib.collections import LineCollection
    tb.open(msname)
    
    data=None
    
    spectrum = ['data_spectrum', 'corr_spectrum', 'resdata_spectrum', 'model_spectrum']
    # Spectrum of chosen column. Make a linecollection out of each row in the MS.
    if myplot in spectrum:
        if myplot=='data_spectrum':
            data = tb.getcol('DATA')
        if myplot=='corr_spectrum':
            data = tb.getcol('CORRECTED_DATA')
        if myplot=='resdata_spectrum':
            data = tb.getcol('DATA') - tb.getcol('MODEL_DATA') 
        if myplot=='rescorr_spectrum':
            data = tb.getcol('CORRECTED_DATA') - tb.getcol('MODEL_DATA') 
        if myplot=='model_spectrum':
            data = tb.getcol('MODEL_DATA')
  
    return data

def get_continuum_data(msname:str, myplot:str, chan_list=list)->'numpy.array':
    '''
        Return continuum (only) data as numpy array

    Args: 
       msname (string): name of ms file
       myplot (str): type of plot
       chan_list (list): list of channels containing a peak

    Returns:
         peak, chans

    '''

    
    baseline = None
    mask = []
    
    # Retrieve full data column from measurement file
    data = get_spectrum_data(msname, myplot)
    
    # Build full channel list
    x = np.zeros((data.shape[2],data.shape[1]),'int')
    for chan in range(0, data.shape[1]):
        x[:,chan] = chan
    
    # Parse data column and append only background
    for lower, upper in chan_list:
        for chan in range(lower, upper+1):
            mask.append(chan)
    
    # Apply channel mask to get baseline
    baseline = np.delete(data, mask, axis=1)
    chans = np.delete(x, mask, axis=1)
            
    return baseline, chans[0]

def get_peak_data(msname:str, myplot:str, chan_list=list)->'numpy.array':
    '''
    Return peak data (only) as numpy array

    Args: 
       msname (string): name of ms file
       myplot (str): type of plot
       chan_list (list): list of channels containing a peak

    Return:
         peaks, chans
    
    '''
    
    
    peak = []
    chans = []
    
    # Retrieve full data column from measurement file
    data = get_spectrum_data(msname, myplot)
    
    # Parse data column and append only background
    for lower, upper in chan_list:
        if len(peak) == 0:
            peak = data[:, lower:upper+1, :]
            chans = np.linspace(lower, upper, upper - lower + 1)
        else:
            peak = np.append(peak, data[:, lower:upper+1, :], axis=1)
            chans = np.append(chans, np.linspace(lower, upper, upper - lower + 1), axis=0)
                
    return peak, chans

def plot_baseline_histogram(msname:str, myplot:str, chan_list:list):
    '''
       Plot histogram of continuum data only.
    '''
    
    baseline, _ = get_continuum_data(
        msname=msname, 
        myplot='data_spectrum', 
        chan_list=[chan_list]
    )
    
    npl = baseline.shape[0]
    
    fig, ax = plt.subplots(npl, 1, figsize=(30, 20))
    
    bins = 100
    
    for polar in range(npl):
        collapsed = baseline[polar, :, :].flatten()
        
        # Histogram of corrected background: real
        _, hist_bins, _ = ax[polar].hist(collapsed.real, bins, color='#f0b27a', density=True)
        ax[polar].set_title("Corrected baseline (real): polarization: {}".format(polar))
        
        mu, sigma = scipy.stats.norm.fit(collapsed.real)
        fit_line = scipy.stats.norm.pdf(hist_bins, mu, sigma)
        
        textstr = '\n'.join((
            r'$\mu=%.3f$' % (mu, ),
            r'$\sigma=%.3f$' % (sigma, )))
        
        props = dict(boxstyle='round', facecolor='#f0b27a', alpha=0.5)
        ax[polar].text(0.05, 0.95, textstr, 
                          transform=ax[polar].transAxes, fontsize=16,
                          verticalalignment='top', bbox=props)
        
        ax[polar].plot(hist_bins, fit_line, color='black', linewidth=2)
        
        
    plt.show()
    
def clean_measurement_files(prefix=""):
    '''
    Parse directory and remove measurement files.

    '''
    
    for file in glob.glob('{0}/{1}*.ms'.format(os.getcwd(), prefix)):
        try:
            shutil.rmtree(file)
        except Exception as error:
            print('Failed to remove {0}: {1}'.format(file, error))
            
            
def compare_visstat(uncorr:str, corr=str)->None:
    '''
    Returns table containing visstat info for both corrected and uncorrected ms files.
    '''
    
    uncorr_stats = visstat(
        vis=uncorr,
        axis='amp',
        datacolumn='data',
        spw='',
        field='',
        correlation='XX'
    )
    
    corr_stats = visstat(
        vis=corr,
        axis='amp',
        datacolumn='data',
        spw='',
        field='',
        correlation='XX'
    )
    
    assert uncorr_stats['DATA_DESC_ID=0'].keys() == corr_stats['DATA_DESC_ID=0'].keys(), 'Keys don\'t match'
    
    table = PrettyTable()
    
    table.field_names = ['Statistic', 'Uncorrected', 'Corrected']
    
    for key, val in uncorr_stats['DATA_DESC_ID=0'].items():
        table.add_row(
            [
                key, 
                uncorr_stats['DATA_DESC_ID=0'][key], 
                corr_stats['DATA_DESC_ID=0'][key]
            ]
        )
        
    return table

def make_spectrum_plotly(msname:list, myplot:str, fitline=list):
    '''
    Spectrum plot utility function using plotly visualization library.
    '''

    
    fig = make_subplots(
        rows=len(msname), cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03
    )
    
    for i, name in enumerate(msname):
        spectrum = get_spectrum_data(
            msname=name, 
            myplot=myplot, 
        )
        
        data = np.mean(spectrum[0, :, :].real, axis=1)
        std = np.std(spectrum[0, :, :].real, axis=1)
        chans = np.linspace(1, data.shape[0], data.shape[0])

        fig.add_trace(
            go.Scatter(
                x=chans, 
                y=data,
                mode='lines+markers',
                name='spectrum: ' + str(i+1),
                marker = dict(
                    size=10, 
                )
            ), row=i+1, col=1
        )
        
        if fitline is not None:
            fit_spectrum = get_spectrum_data(
                msname=fitline[i], 
                myplot='model_spectrum'
            )
                
            fit_poly = np.mean(fit_spectrum[0, :, :].real, axis=1)
                
            fig.add_trace(
                go.Scatter(
                x=chans, 
                y=fit_poly,
                mode='lines',
                name='fit: ' + str(i+1),
                line=dict(
                    color='#000000',
                )
            ), row=i+1, col=1
        )

        
    fig.update_layout(
        height=1000,
        showlegend=True,
        title_text="Spectrum+Continuum for each Order (n)",
    )
    
    fig.show()

def make_synopsis_plotly(uncorr:str, corr:str, chan_list:list, title="Analysis Overview"):
    '''
    Utility function returning syunopsis plots and visstat comparison table for corrected ms files.
    '''
    
    import plotly.figure_factory as ff
    import plotly.graph_objects as go

    from plotly.subplots import make_subplots

    # Retrieve corrected baseline
    baseline, _ = get_continuum_data(
        msname=corr,
        myplot='data_spectrum',
        chan_list=[chan_list]
    )

    # Retrieve uncorrected spectrum
    uncorr_spectrum = get_spectrum_data(
        msname=uncorr,
        myplot='data_spectrum',
    )

    # Retrieve corrected spectrum
    corr_spectrum = get_spectrum_data(
        msname=corr,
        myplot='data_spectrum',
    )

    # Calculate visstat values
    uncorr_stats = visstat(
        vis=uncorr,
        axis='real',
        datacolumn='data',
        spw='',
        field='',
        correlation='XX'
    )

    corr_stats = visstat(
        vis=corr,
        axis='real',
        datacolumn='data',
        spw='',
        field='',
        correlation='XX'
    )

    # Visstat results must having matching dimensions
    assert uncorr_stats['DATA_DESC_ID=0'].keys() == corr_stats['DATA_DESC_ID=0'].keys(), 'Keys don\'t match'

    # Make list of visstat values
    visstat_keys = []
    visstat_corrected_values = []
    visstat_uncorrected_values = []

    for key, _ in uncorr_stats['DATA_DESC_ID=0'].items():
                visstat_keys.append(key),
                visstat_uncorrected_values.append(uncorr_stats['DATA_DESC_ID=0'][key]),
                visstat_corrected_values.append(corr_stats['DATA_DESC_ID=0'][key])

    uncorr_data = np.mean(uncorr_spectrum[0, :, :].real, axis=1)
    chans = np.linspace(1, uncorr_data.shape[0], uncorr_data.shape[0])

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=False,
        vertical_spacing=0.03,
        specs=[
            [{"type": "scatter"}],
            [{"type": "scatter"}],
            [{"type": "histogram"}],
            [{"type": "table"}]
        ]
    )

    fig.add_trace(
        go.Scatter(
            x=chans,
            y=uncorr_data,
            mode="lines+markers",
            name='Uncorrected Spectrum'
        ),
        row=1, col=1
    )

    corr_data = np.mean(corr_spectrum[0, :, :].real, axis=1)

    fig.add_trace(
        go.Scatter(
            x=chans,
            y=corr_data,
            mode="lines+markers",
            name='Corrected Spectrum',
            marker=dict(
                color=' #1ABC9C'
            )
        ),
        row=2, col=1
    )

    collapsed = baseline[0, :, :].real.flatten()

    dist = ff.create_distplot(
        [collapsed],
        curve_type='normal',
        group_labels=['continuum']
    )

    fig.add_trace(
        go.Histogram(
            x=dist.data[0]['x'],
            histnorm='probability density',
            name='Corrected Continuum',
            marker=dict(
                color='#FF9800 ',
                opacity=0.8
            )
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dist.data[1]['x'],
            y=dist.data[1]['y'],
            name='Normal Fit',
            line=dict(
                    color='#000000',
                ),
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Table(
            header=dict(
                values=['Statistic', 'Uncorrected', 'Corrected'],
                font=dict(size=10),
                align="center"
            ),
            cells=dict(
                values=[
                    visstat_keys,
                    visstat_corrected_values,
                    visstat_corrected_values
                ],
                align = "left")
        ), row=4, col=1
    )

    fig.update_layout(
        height=1000,
        showlegend=True,
        title_text=title,
    )

    fig['layout'].update(
        annotations=[
            dict(
                x=0.9, y=0.9,
                xref='x3 domain',
                yref='y3 domain',
                text='mean: {0}<br>stdev: {1}<br>skew: {2}'.format(
                    round(collapsed.mean(), 5),
                    round(collapsed.std(), 5),
                    round(scipy.stats.skew(collapsed), 5)
                ),
                showarrow=False,
                borderwidth=1,
                bordercolor='black',
                bgcolor='white',
                ax=10,
                ay=70
            )
        ]
    )
    fig['layout']['yaxis']['range']=[-0.1, 1.2*uncorr_data.max()]
    fig['layout']['yaxis2']['range']=[-0.1, 1.2*uncorr_data.max()]

    fig.show() 
