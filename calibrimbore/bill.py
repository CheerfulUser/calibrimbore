import pysynphot as S
import numpy as np
from astroquery.vizier import Vizier
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord, Angle


import os
package_directory = os.path.dirname(os.path.abspath(__file__)) + '/'


import astropy.table as at
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import astropy.table as at

import pandas as pd
from glob import glob
from copy import deepcopy
from scipy.optimize import minimize
from astropy.stats import sigma_clip
from .sigmacut import calcaverageclass

from scipy.interpolate import UnivariateSpline

cas_id = os.environ.get('CASJOBS_WSID')
cas_pwd = os.environ.get('CASJOBS_PW')

fig_width_pt = 240.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inches
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches


g = np.loadtxt(package_directory + 'data/ps1_bands/ps1_g.dat')
r = np.loadtxt(package_directory + 'data/ps1_bands/ps1_r.dat')
i = np.loadtxt(package_directory + 'data/ps1_bands/ps1_i.dat')
z = np.loadtxt(package_directory + 'data/ps1_bands/ps1_z.dat')
y = np.loadtxt(package_directory + 'data/ps1_bands/ps1_y.dat')

ps1_bands = {'g': S.ArrayBandpass(g[:,0],g[:,1]),
             'r': S.ArrayBandpass(r[:,0],r[:,1]),
             'i': S.ArrayBandpass(i[:,0],i[:,1]),  
             'z': S.ArrayBandpass(z[:,0],z[:,1]),
             'y': S.ArrayBandpass(y[:,0],y[:,1])}

g = np.loadtxt(package_directory + 'data/skymappaer_bands/sm_g.dat')
r = np.loadtxt(package_directory + 'data/skymappaer_bands/sm_r.dat')
i = np.loadtxt(package_directory + 'data/skymappaer_bands/sm_i.dat')
z = np.loadtxt(package_directory + 'data/skymappaer_bands/sm_z.dat')
v = np.loadtxt(package_directory + 'data/skymappaer_bands/sm_v.dat')
u = np.loadtxt(package_directory + 'data/skymappaer_bands/sm_u.dat')

sm_bands = {'g': S.ArrayBandpass(g[:,0],g[:,1]),
             'r': S.ArrayBandpass(r[:,0],r[:,1]),
             'i': S.ArrayBandpass(i[:,0],i[:,1]),  
             'z': S.ArrayBandpass(z[:,0],z[:,1]),
             'v': S.ArrayBandpass(v[:,0],v[:,1]),
             'u': S.ArrayBandpass(u[:,0],u[:,1])}

def get_pb_zpt(pb, reference='AB', model_mag=None):
    """
    Determines a passband zeropoint for synthetic photometry, given a reference
    standard and its model magnitude in the passband

    Parameters
    ----------
    pb : :py:class:`pysynphot.ArrayBandpass` or :py:class:`pysynphot.obsbandpass.ObsModeBandpass`
        The passband data.
        Must have ``dtype=[('wave', '<f8'), ('throughput', '<f8')]``
    reference : str, optional
        The name of the reference spectrophotometric standard to use to determine the passband zeropoint.
        'AB' or 'Vega' (default: 'AB')
    model_mag : float, optional
        The magnitude of the reference spectrophotometric standard in the passband.
        default = None

    Returns
    -------
    pbzp : float
        passband AB zeropoint

    Raises
    ------
    RuntimeError
        If the passband and reference standard do not overlap
    ValueError
        If the reference model_mag is invalid

    See Also
    --------
    :py:func:`source_synphot.passband.synflux`
    :py:func:`source_synphot.passband.synphot`
    """

    # setup the photometric system by defining the standard and corresponding magnitude system
    if reference.lower() not in ('vega', 'ab'):
        message = 'Do not understand mag system reference standard {}. Must be AB or Vega'.format(reference)
        raise RuntimeError(message)
    if reference.lower() == 'ab':
        refspec = S.FlatSpectrum(3631, fluxunits='jy')
        mag_type= 'abmag'
    else:
        refspec = S.Vega
        mag_type= 'vegamag'

    refspec.convert('flam')

    if model_mag is None:
        ob = S.Observation(refspec, pb)
        model_mag = ob.effstim(mag_type)

    synphot_mag = -2.5*np.log10(synflux(refspec, pb))
    outzp = model_mag - synphot_mag
    return outzp


def synflux(spec, pb):
    """
    Compute the synthetic flux of spectrum ``spec`` through passband ``pb``

    Parameters
    ----------
    spec : :py:class:`pysynphot.ArraySpectrum`
        The spectrum. Must have ``dtype=[('wave', '<f8'), ('flux', '<f8')]``
    pb : :py:class:`pysynphot.ArrayBandpass` or :py:class:`pysynphot.obsbandpass.ObsModeBandpass`
        The passband data.
        Must have ``dtype=[('wave', '<f8'), ('throughput', '<f8')]``

    Returns
    -------
    flux : float
        The normalized flux of the spectrum through the passband

    Notes
    -----
        The passband is assumed to be dimensionless photon transmission
        efficiency.

        Routine is intended to be a mch faster implementation of
        :py:meth:`pysynphot.observation.Observation.effstim`, since it is called over and
        over by the samplers as a function of model parameters.

        Uses :py:func:`numpy.trapz` for interpolation.
    """
    overlap = pb.check_overlap(spec)
    flux = spec.sample(pb.wave)
    n = np.trapz(flux*pb.wave*pb.throughput, pb.wave)
    d = np.trapz(pb.wave*pb.throughput, pb.wave)
    out = n/d
    return out


def synmag(spec, pb, zp=0.):
    """
    Compute the synthetic magnitude of spectrum ``spec`` through passband ``pb``

    Parameters
    ----------
    spec : :py:class:`pysynphot.ArraySpectrum`
        The spectrum. Must have ``dtype=[('wave', '<f8'), ('flux', '<f8')]``
    pb : :py:class:`pysynphot.ArrayBandpass` or :py:class:`pysynphot.obsbandpass.ObsModeBandpass`
        The passband transmission.
    zp : float, optional
        The zeropoint to apply to the synthetic flux

    Returns
    -------
    mag : float
        The synthetic magnitude of the spectrum through the passband

    See Also
    --------
    :py:func:`source_synphot.passband.synflux`
    """
    flux = synflux(spec, pb)
    m = -2.5*np.log10(flux) + zp
    return m

def get_ps1_region(ra,dec,size=0.2*60**2):
    """
    Get PS1 observations for a region.
    
    ------
    Inputs 
    ------
    ra : `float`
        RA of target
    dec : `float`
        Dec of target
    size : `float`
        Search radius in arcsec, 0.2 deg is a good default 
    
    -------
    Returns
    -------
    final : `pandas Dataframe`
        Table containing all relevant PS1 observations for each object entered 
        with the ra and dec lists.
    """
    if (type(ra) == float) | (type(ra) == np.float64):
        ra = [ra]
    if (type(dec) == float) | (type(dec) == np.float64):
        dec = [dec]
    coords = Table(data=[ra*u.deg,dec*u.deg],names=['_RAJ2000','_DEJ2000'])
    
    Vizier.ROW_LIMIT = -1
    
    catalog = "II/349/ps1"
    print('Querying regions with Vizier')
    result = Vizier.query_region(coords, catalog=[catalog],
                                 radius=Angle(size, "arcsec"))
    no_targets_found_message = ValueError('Either no sources were found in the query region '
                                          'or Vizier is unavailable')
    if result is None:
        raise no_targets_found_message
    elif len(result) == 0:
        raise no_targets_found_message
    result = result[catalog].to_pandas()
    r = deepcopy(result)
    final = pd.DataFrame(data=np.zeros(len(r)),columns=['temp'])
    final['ra'] = np.nan
    final['dec'] = np.nan
    final['g'] = np.nan; final['r'] = np.nan; final['i'] = np.nan; 
    final['z'] = np.nan; final['y'] = np.nan
    final['g_e'] = np.nan; final['r_e'] = np.nan; final['i_e'] = np.nan; 
    final['z_e'] = np.nan; final['y_e'] = np.nan

    final['g'] = r['gmag'].values; final['r'] = r['rmag'].values; final['i'] = r['imag'].values
    final['z'] = r['zmag'].values; final['y'] = r['ymag'].values

    final['g_e'] = r['e_gmag'].values; final['r_e'] = r['e_rmag'].values; final['i_e'] = r['e_imag'].values
    final['z_e'] = r['e_zmag'].values; final['y_e'] = r['e_ymag'].values
    final['ra'] = r['RAJ2000'].values; final['dec'] = r['DEJ2000'].values
    final = final.drop(['temp'], axis=1)
    return final

def get_skymapper_region(ra,dec,size=0.2*60**2):
    """
    Get PS1 observations for a region.
    
    ------
    Inputs 
    ------
    ra : `float`
        RA of target
    dec : `float`
        Dec of target
    size : `float`
        Search radius in arcsec, 0.2 deg is a good default 
    
    -------
    Returns
    -------
    final : `pandas Dataframe`
        Table containing all relevant PS1 observations for each object entered 
        with the ra and dec lists.
    """
    if (type(ra) == float) | (type(ra) == np.float64):
        ra = [ra]
    if (type(dec) == float) | (type(dec) == np.float64):
        dec = [dec]
    coords = Table(data=[ra*u.deg,dec*u.deg],names=['_RAJ2000','_DEJ2000'])
    
    Vizier.ROW_LIMIT = -1
    
    catalog = "II/358/smss"
    print('Querying regions with Vizier')
    result = Vizier.query_region(coords, catalog=[catalog],
                                 radius=Angle(size, "arcsec"))
    no_targets_found_message = ValueError('Either no sources were found in the query region '
                                          'or Vizier is unavailable')
    if result is None:
        raise no_targets_found_message
    elif len(result) == 0:
        raise no_targets_found_message
    result = result[catalog].to_pandas()
    r = deepcopy(result)
    final = pd.DataFrame(data=np.zeros(len(r)),columns=['temp'])
    final['ra'] = np.nan
    final['dec'] = np.nan
    final['g'] = np.nan; final['r'] = np.nan; final['i'] = np.nan; 
    final['z'] = np.nan; final['u'] = np.nan
    final['g_e'] = np.nan; final['r_e'] = np.nan; final['i_e'] = np.nan; 
    final['z_e'] = np.nan; final['u_e'] = np.nan

    final['g'] = r['gPSF'].values; final['r'] = r['rPSF'].values; final['i'] = r['iPSF'].values
    final['z'] = r['zPSF'].values; final['u'] = r['uPSF'].values

    final['g_e'] = final['g'].values * np.nan; final['r_e'] = final['g'].values * np.nan 
    final['i_e'] = final['g'].values * np.nan; final['z_e'] = final['g'].values * np.nan 
    final['y_e'] = r['e_ymag'].values

    final['ra'] = r['RAICRS'].values; final['dec'] = r['DEICRS'].values
    final = final.drop(['temp'], axis=1)

    return final



def query_casjob_ps1(ra,dec,size=3):
    import mastcasjobs
    query = """SELECT o.objID,
                o.raMean as ra, o.decMean as dec, o.raMeanErr, o.decMeanErr,
                o.gMeanPSFMag as g, o.gMeanPSFMagErr as g_e,
                o.rMeanPSFMag as r, o.rMeanPSFMagErr as r_e,
                o.iMeanPSFMag as i, o.iMeanPSFMagErr as i_e,
                o.zMeanPSFMag as z, o.zMeanPSFMagErr as z_e,
                o.yMeanPSFMag as y, o.yMeanPSFMagErr as y_e,
                o.nDetections, o.ng, o.nr, o.ni, o.nz,o.ny,
                o.gFlags, o.gQfPerfect,
                o.rFlags, o.rQfPerfect,
                o.iFlags, o.iQfPerfect,
                o.zFlags, o.zQfPerfect,
                o.yFlags, o.yQfPerfect
                FROM dbo.fGetNearbyObjEq({},{},{}/60.0) as x
                JOIN MeanObjectView o on o.ObjID=x.ObjId
                LEFT JOIN StackObjectAttributes AS soa ON soa.objID = x.objID
                WHERE o.nDetections>5
                AND soa.primaryDetection>0
                AND o.gQfPerfect>0.85 and o.rQfPerfect>0.85 and o.iQfPerfect>0.85 and o.zQfPerfect>0.85
                """.format(ra,dec,size)# % {'ra':ra,'dec':dec,'size':size}
    jobs = mastcasjobs.MastCasJobs(userid=cas_id, password=cas_pwd, context="PanSTARRS_DR2")
    results = jobs.quick(query, task_name="python cone search")
    if len(results) == 0:
        empty = np.ones(len(results.keys())) * -999
        results.add_row(vals=empty)
    results = results.to_pandas()
    results = results.replace(to_replace=-999,value=np.nan)
    return results

def ps1_casjobs(ra,dec,size=3):
    
    for i in range(len(ra)):
        tab = query_casjob_ps1(ra[i],dec[i],size)
        if i == 0:
            result = tab
        else:
            result = result.append(tab, ignore_index=True)
    #ind = result.values < -900
    #result.iloc[ind] = np.nan
    return result

def get_ps1(ra,dec,size=3):
    """
    Get PS1 observations for a list of coordinates.
    
    ------
    Inputs 
    ------
    ra : `list`/`array`
        RA of target
    dec : `list`/`array`
        Dec of target
    size : `float`
        Search radius in arcsec, 3 arcsec is a good default.
    
    -------
    Returns
    -------
    final : `pandas Dataframe`
        Table containing all relevant PS1 observations for each object entered 
        with the ra and dec lists.
    """
    if (type(ra) == float) | (type(ra) == np.float64):
        ra = [ra]
    if (type(dec) == float) | (type(dec) == np.float64):
        dec = [dec]
    coords = Table(data=[ra*u.deg,dec*u.deg],names=['_RAJ2000','_DEJ2000'])
    
    Vizier.ROW_LIMIT = -1
    
    catalog = "II/349/ps1"
    print('Querying regions with Vizier')
    result = Vizier.query_region(coords, catalog=[catalog],
                                 radius=Angle(size, "arcsec"))
    no_targets_found_message = ValueError('Either no sources were found in the query region '
                                          'or Vizier is unavailable')
    if result is None:
        raise no_targets_found_message
    elif len(result) == 0:
        raise no_targets_found_message
    result = result[catalog].to_pandas()

    targets = coords.to_pandas()

    dist = np.zeros((len(targets),len(result)))

    dist = ((targets['_RAJ2000'].values[:,np.newaxis] - result['RAJ2000'].values[np.newaxis,:])**2 +
            (targets['_DEJ2000'].values[:,np.newaxis] - result['DEJ2000'].values[np.newaxis,:])**2)

    min_ind = np.argmin(dist,axis=1)
    ind = np.nanmin(dist,axis=1) <= size

    min_ind2 = np.argmin(dist,axis=0)

    r = deepcopy(result.iloc[min_ind])

    final = deepcopy(targets)
    final['ra'] = np.nan
    final['dec'] = np.nan
    final['g'] = np.nan; final['r'] = np.nan; final['i'] = np.nan; 
    final['z'] = np.nan; final['u'] = np.nan
    final['g_e'] = np.nan; final['r_e'] = np.nan; final['i_e'] = np.nan; 
    final['z_e'] = np.nan; final['u_e'] = np.nan

    final['g'].iloc[ind] = r['gPSF'].values[ind]; final['r'].iloc[ind] = r['rPSF'].values[ind] 
    final['i'].iloc[ind] = r['iPSF'].values[ind]; final['z'].iloc[ind] = r['zPSF'].values[ind]
    final['y'].iloc[ind] = r['yPSF'].values[ind]

    final['g_e'].iloc[ind] = r['gPSF'].values[ind]*np.nan; final['r_e'].iloc[ind] = r['gPSF'].values[ind]*np.nan
    final['i_e'].iloc[ind] = r['gPSF'].values[ind]*np.nan; final['z_e'].iloc[ind] = r['gPSF'].values[ind]*np.nan
    final['u_e'].iloc[ind] = r['gPSF'].values[ind]*np.nan

    final['ra'].iloc[ind] = r['RAJ2000'].values[ind]; final['dec'].iloc[ind] = r['DEJ2000'].values[ind]
    
    return final 


def get_skymapper(ra,dec,size=3):
    """
    Get PS1 observations for a list of coordinates.
    
    ------
    Inputs 
    ------
    ra : `list`/`array`
        RA of target
    dec : `list`/`array`
        Dec of target
    size : `float`
        Search radius in arcsec, 3 arcsec is a good default.
    
    -------
    Returns
    -------
    final : `pandas Dataframe`
        Table containing all relevant PS1 observations for each object entered 
        with the ra and dec lists.
    """
    if (type(ra) == float) | (type(ra) == np.float64):
        ra = [ra]
    if (type(dec) == float) | (type(dec) == np.float64):
        dec = [dec]
    coords = Table(data=[ra*u.deg,dec*u.deg],names=['_RAJ2000','_DEJ2000'])
    
    Vizier.ROW_LIMIT = -1
    
    catalog = "II/358/smss"
    print('Querying regions with Vizier')
    result = Vizier.query_region(coords, catalog=[catalog],
                                 radius=Angle(size, "arcsec"))
    no_targets_found_message = ValueError('Either no sources were found in the query region '
                                          'or Vizier is unavailable')
    if result is None:
        raise no_targets_found_message
    elif len(result) == 0:
        raise no_targets_found_message
    result = result[catalog].to_pandas()

    targets = coords.to_pandas()

    dist = np.zeros((len(targets),len(result)))

    dist = ((targets['_RAJ2000'].values[:,np.newaxis] - result['RAJ2000'].values[np.newaxis,:])**2 +
            (targets['_DEJ2000'].values[:,np.newaxis] - result['DEJ2000'].values[np.newaxis,:])**2)

    min_ind = np.argmin(dist,axis=1)
    ind = np.nanmin(dist,axis=1) <= size

    min_ind2 = np.argmin(dist,axis=0)

    r = deepcopy(result.iloc[min_ind])

    final = deepcopy(targets)
    final['ra'] = np.nan
    final['dec'] = np.nan
    final['g'] = np.nan; final['r'] = np.nan; final['i'] = np.nan; 
    final['z'] = np.nan; final['y'] = np.nan
    final['g_e'] = np.nan; final['r_e'] = np.nan; final['i_e'] = np.nan; 
    final['z_e'] = np.nan; final['y_e'] = np.nan

    final['g'].iloc[ind] = r['gmag'].values[ind]; final['r'].iloc[ind] = r['rmag'].values[ind]; final['i'].iloc[ind] = r['imag'].values[ind]
    final['z'].iloc[ind] = r['zmag'].values[ind]; final['y'].iloc[ind] = r['ymag'].values[ind]

    final['g_e'].iloc[ind] = r['e_gmag'].values[ind]; final['r_e'].iloc[ind] = r['e_rmag'].values[ind]; final['i_e'].iloc[ind] = r['e_imag'].values[ind]
    final['z_e'].iloc[ind] = r['e_zmag'].values[ind]; final['y_e'].iloc[ind] = r['e_ymag'].values[ind]
    final['ra'].iloc[ind] = r['RAJ2000'].values[ind]; final['dec'].iloc[ind] = r['DEJ2000'].values[ind]
    return final 



# Tools to use the Tonry 2012 PS1 color splines to fit extinction

def Tonry_clip(Colours):
    """
    Use the Tonry 2012 PS1 splines to sigma clip the observed data.
    """
    tonry = np.loadtxt(package_directory + '/data/Tonry_splines.txt')
    X = 'r-i'
    Y = 'g-r'
    x = Colours['obs r-i'][0,:]
    mx = tonry[:,0]
    y = Colours['obs g-r'][0,:]
    my = tonry[:,1]
    # set up distance matrix
    xx = x[:,np.newaxis] - mx[np.newaxis,:]
    yy = y[:,np.newaxis] - my[np.newaxis,:]
    # calculate distance
    dd = np.sqrt(xx**2 + yy**2)
    # return min values for the observation axis
    mins = np.nanmin(dd,axis=1)
    # Sigma clip the distance data
    ind = np.isfinite(mins)
    sig = sigma_mask(mins[ind])
    # return sigma clipped mask
    ind[ind] = ~sig
    return ind

def Tonry_residual(Colours):
    """
    Calculate the residuals of the observed data from the Tonry et al 2012 PS1 splines.
    """
    tonry = np.loadtxt(package_directory + 'data/Tonry_splines.txt')
    X = 'r-i'
    Y = 'g-r'
    x = Colours['obs ' + X][0,:]
    mx = tonry[:,0]
    y = Colours['obs ' + Y][0,:]
    my = tonry[:,1]
    # set up distance matrix
    xx = x[:,np.newaxis] - mx[np.newaxis,:]
    yy = y[:,np.newaxis] - my[np.newaxis,:]
    # calculate distance
    dd = np.sqrt(xx**2 + yy**2)
    # return min values for the observation axis
    mingr = np.nanmin(dd,axis=1)
    return np.nansum(mingr) #+ np.nansum(miniz)

def Tonry_fit(K,Data,Model,Compare):
    """
    Wrapper for the residuals function
    """
    Colours = Make_colours(Data,Model,Compare,Extinction = K,Redden=False, Tonry = True)
    res = Tonry_residual(Colours)
    return res

def Tonry_reduce(Data,plot=False,savename=None):
    '''
    Uses the Tonry et al. 2012 PS1 splines to fit dust and find all outliers.
    '''
    data = deepcopy(Data)
    tonry = np.loadtxt(package_directory + 'data/Tonry_splines.txt')
    compare = np.array([['r-i','g-r']])   
    dat = data
    clips = []
    if len(dat) < 10:
        raise ValueError('No data available')
    for i in range(2):
        if i == 0:
            k0 = 0.01
        else:
            k0 = res.x

        res = minimize(Tonry_fit,k0,args=(dat,tonry,compare),method='Nelder-Mead')
        
        colours = Make_colours(dat,tonry,compare,Extinction = res.x, Tonry = True)
        clip = Tonry_clip(colours)
        clips += [clip]
        dat = dat.iloc[clip]
        
    clips[0][clips[0]] = clips[1]
    if plot:
        orig = Make_colours(dat,tonry,compare,Extinction = 0, Tonry = True)
        colours = Make_colours(dat,tonry,compare,Extinction = res.x, Tonry = True)
        plt.figure(figsize=(1.5*fig_width,1*fig_width))
        #plt.title('Fit to Tonry et al. 2012 PS1 stellar locus')
        plt.plot(orig['obs r-i'].flatten(),orig['obs g-r'].flatten(),'C1+',alpha=0.5,label='Raw')
        plt.plot(colours['obs r-i'].flatten(),colours['obs g-r'].flatten(),'C0.',alpha=0.5,label='Corrected')
        plt.plot(colours['mod r-i'].flatten(),colours['mod g-r'].flatten(),'k-',label='Model')
        plt.xlabel('$r-i$',fontsize=15)
        plt.ylabel('$g-r$',fontsize=15)
        plt.text(1, 0.25, '$E(B-V)={}$'.format(str(np.round(res.x[0],3))))
        plt.legend()
        if savename is not None:
            plt.savefig(savename + '_SLR.pdf', bbox_inches = "tight")
    
    return res.x, dat



def sigma_mask(data,error= None,sigma=3,Verbose= False):
    """
    Mask data using the sigmacut class.
    """
    if type(error) == type(None):
        error = np.zeros(len(data))
    
    calcaverage = calcaverageclass()
    calcaverage.calcaverage_sigmacutloop(data,Nsigma=sigma
                                         ,median_firstiteration=True,saveused=True)
    if Verbose:
        print("mean:%f (uncertainty:%f)" % (calcaverage.mean,calcaverage.mean_err))
    return calcaverage.clipped



def Get_lcs(X,Y,K,Colours,fitfilt = ''):
    """
    Make the colour combinations
    """
    keys = np.array(list(Colours.keys()))

    xind = 'mod ' + X == keys
    x = Colours[keys[xind][0]]
    yind = 'mod ' + Y == keys
    y = Colours[keys[yind][0]]

    locus = np.array([x,y])

    xind = 'obs ' + X == keys
    x = Colours[keys[xind][0]]
    yind = 'obs ' + Y == keys
    y = Colours[keys[yind][0]]
    c1,c2 = X.split('-')
    c3,c4 = Y.split('-')
    # parameters
    ob_x = x.copy() 
    ob_y = y.copy() 

    if c1 == fitfilt: ob_x[0,:] += K
    if c2 == fitfilt: ob_x[0,:] -= K

    if c3 == fitfilt: ob_y[0,:] += K
    if c4 == fitfilt: ob_y[0,:] -= K
    return ob_x, ob_y, locus


def Dist_tensor(X,Y,K,Colours,fitfilt='',Tensor=False,Plot = False):
    """
    Calculate the distance of sources in colour space from the model stellar locus.
    
    ------
    Inputs
    ------
    X : `str`
        string containing the colour combination for the X axis 
    Y : `str`
        string containing the colour combination for the Y axis 
    K : `str`
        Not sure...
    Colours : `dict`
        dictionary of colour combinations for all sources 
    fitfilt : `str`
         Not used...
     Tensor : `bool`
        if true this returns the distance tensor instead of the total sum
    Plot : `bool`
        if true this makes diagnotic plots

    -------
    Returns
    -------
    residual : `float`
        residuals of distances from all points to the model locus. 
    """
    ob_x, ob_y, locus = Get_lcs(X,Y,K,Colours,fitfilt)
    
    ind = np.where((Colours['obs g-r'][0,:] <= .8) & (Colours['obs g-r'][0,:] >= 0.2))[0]
    indo = np.where((Colours['obs g-r'][0,:] <= .8) & (Colours['obs g-r'][0,:] >= 0.2))
    ob_x = ob_x[:,ind]
    ob_y = ob_y[:,ind]
    
    
    if Plot:
        plt.figure()
        plt.title(X + ' ' + Y)
        plt.plot(ob_x[0,:],ob_y[0,:],'.')
        plt.plot(locus[0,:],locus[1,:])

    x = np.zeros((ob_x.shape[1],locus.shape[1])) + ob_x[0,:,np.newaxis]
    x -= locus[0,np.newaxis,:]
    y = np.zeros((ob_y.shape[1],locus.shape[1])) + ob_y[0,:,np.newaxis]
    y -= locus[1,np.newaxis,:]

    dist_tensor = np.sqrt(x**2 + y**2)
    
    if len(dist_tensor[np.isfinite(dist_tensor)]) > 1:
        minind = np.nanargmin(abs(dist_tensor),axis=1)
        mindist = np.nanmin(abs(dist_tensor),axis=1)
        sign = (ob_y[0,:] - locus[1,minind])
        sign = sign / abs(sign)

        eh = mindist * sign
    
        proj_err = Dot_prod_error(ob_x,ob_y,locus[:,minind])
        
        if Tensor:
            return eh
        if len(mindist) > 0:
            residual = np.nansum(abs(mindist)) #/ proj_err)
        else:
            residual = np.inf
    else:
        if Tensor:
            return []
        residual = np.inf

    cut_points = len(indo) - len(ind)
    return residual + cut_points * 100


def Make_colours(Data, Model, Compare, Extinction = 0, Redden = False,Tonry=False):
    """
    Make dictionaries of colour combinations used in stellar locus regression code.

    ------
    Inputs
    ------
    Data : pandas DataFrame
        Contains the PS1 photometry in the required format
    Model : pandas DataFrame
        Contains the model PS1 photometry
    Compare : `list`
        List of tuples containing the colours to compare.
    Extinction : `float`
        Extinction in terms of `E(B-V)` to be applied to the photometry
    Redden : `bool`
        Switch to apply the extinction to the model data
    Tonry : `bool`
        Switch to comparing against the Tonry 2012 PS1 stellar locus.
        This is used for determining the extinction

    -------
    Returns
    -------
    colours : `dict`
        Dictionary containing the model and observed colours used for 
        calibrating data through stellar locus regression.
    """
    R = {'g': 3.61562687, 'r':2.58602003, 'i':1.90959054, 
         'z':1.50168735, 'y': 1.25340149}
    colours = {}
    for x,y in Compare:
        colours['obs ' + x] = np.array([Data[x.split('-')[0]].values - Data[x.split('-')[1]].values,
                                        Data[x.split('-')[0]+'_e'].values - Data[x.split('-')[1]+'_e'].values])
        colours['obs ' + y] = np.array([Data[y.split('-')[0]].values - Data[y.split('-')[1]].values,
                                        Data[y.split('-')[0]+'_e'].values - Data[y.split('-')[1]+'_e'].values])
        if Tonry:
            colours['mod ' + x] = Model[:,0]
            colours['mod ' + y] = Model[:,1]
            # colour cut to remove weird top branch present in C2
            if (y == 'g-r'):
                ind = colours['obs g-r'][0,:] > 1.4
                colours['obs g-r'][:,ind] = np.nan
                colours['obs r-i'][:,ind] = np.nan
        else:

            xx = Model[x.split('-')[0]] - Model[x.split('-')[1]]
            yy = Model[y.split('-')[0]] - Model[y.split('-')[1]]
            ind = xx.argsort()
            xx = xx[ind]
            yy = yy[ind]
            spl = UnivariateSpline(xx, yy)
            c_range = np.arange(xx[0],0.8,0.01)
            colours['mod ' + x] = c_range
            colours['mod ' + y] = spl(c_range)
        
        if Redden:
            colours['mod ' + x] += Extinction*((R[x.split('-')[0]] - R[x.split('-')[1]]))
            colours['mod ' + y] += Extinction*(R[y.split('-')[0]] - R[y.split('-')[1]])
        else:
            colours['obs ' + x] -= Extinction*((R[x.split('-')[0]] - R[x.split('-')[1]]))
            colours['obs ' + y] -= Extinction*(R[y.split('-')[0]] - R[y.split('-')[1]])
    return colours 

