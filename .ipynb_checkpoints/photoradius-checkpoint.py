# Install a pip package in the current Jupyter kernel
import sys
import os
sys.path.append('../')

### General
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import astropy
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm

### Tools
import WD_models
from astropy.coordinates import SkyCoord
import astropy.units as u
from dustmaps.bayestar import BayestarQuery

#corv.sdss.make_catalogs()

c = 2.998e10
k = 1.38e-16
h = 6.626e-27
halpha = 6564.61
hbeta = 4862.68
hgamma = 4341.68
hdelta = 4102.89
speed_light = 299792458 #m/s
radius_sun = 6.957e8
mass_sun = 1.9884e30
newton_G = 6.674e-11
pc_to_m = 3.086775e16

bsq = BayestarQuery()


def find_radius(l, b, parallax, obs_mag, e_obs_mag, floor_error = 0.03, deredden = True, make_plot = False, vary_logg = False, p0 = [10000, 8, 0.01, 100], bsq = None):    
    dist = 1000 / parallax
    
    if deredden:
        coords = [SkyCoord(frame="galactic", l=l*u.deg, b=b*u.deg, distance = (dist) * u.kpc) for i in range(len(dist))]
        ebv = np.array([bsq.query(coords[i]) for i in range(len(coords))])
    
        obs_mag[0] = obs_mag[0] - ebv * 3.518
        obs_mag[1] = obs_mag[1] - ebv * 2.617
        obs_mag[2] = obs_mag[2] - ebv * 1.971
        obs_mag[3] = obs_mag[3] - ebv * 1.549
        obs_mag[4] = obs_mag[4] - ebv * 1.263
        
    font_model = WD_models.load_model('f', 'f', 'f', 'H', HR_bands = ['Su-Sg', 'Su'])

    g_acc = (10**font_model['logg'])/100
    rsun = np.sqrt(font_model['mass_array'] * mass_sun * newton_G / g_acc) / radius_sun
    logg_teff_to_rsun = WD_models.interp_xy_z_func(x = font_model['logg'], y = 10**font_model['logteff'],
                                                z = rsun, interp_type = 'linear')
    
    fitsed = WD_models.FitSED(to_flux = False, atm_type = 'H', bands = ['Su', 'Sg', 'Sr', 'Si', 'Sz'])
    
    def mag_to_flux(mag):
        return 10 ** ((mag + 48.6)/ -2.5) 
        
    def get_model_flux(params):
        
        teff, logg, radius, distance = params['teff'], params['logg'], params['radius'], params['distance']
        
        model_mag = fitsed.model_sed(teff, logg, plx = 100)
        model_flux = mag_to_flux(model_mag)
        
        rsun = logg_teff_to_rsun(logg, teff)
        corr_radius = rsun * radius_sun
        corr_distance = 10 * pc_to_m
        
        corr_model_flux = model_flux / (4 * np.pi * (corr_radius / corr_distance)**2)
        
        radius = radius * radius_sun # Rsun to meter
        distance = distance * pc_to_m # Parsec to meter
         
        flux = corr_model_flux * ( 4 * np.pi * (radius / distance)**2 )
        
        return flux
        
    
    def residual(params, obs_flux = None, e_obs_flux = None):
        model_flux = get_model_flux(params)
    
        chisquare = ((model_flux - obs_flux) / e_obs_flux)**2
        return chisquare
    
    obs_flux = mag_to_flux(obs_mag)
    
    e_obs_mag = np.sqrt(e_obs_mag**2 + floor_error**2)
    e_obs_flux = e_obs_mag * obs_flux
    
    params = lmfit.Parameters()

    params.add('teff', value = p0[0], min = 3500, max = 55000, vary = True)
    params.add('logg', value = p0[1], min=7.5, max=9, vary=vary_logg)
    params.add('radius', value = p0[2], min = 0.0001, max = 0.05, vary = True)
    params.add('distance', value = p0[3], min = 1, max = 2000, vary = False)
        
    #result = lmfit.minimize(residual, params, kws = dict(obs_flux = obs_flux, e_obs_flux = e_obs_flux), method = 'emcee', steps = 5000, burn = 500, thin = 20, is_weighted = False, progress = False)
    result = lmfit.minimize(residual, params, kws = dict(obs_flux = obs_flux, e_obs_flux = e_obs_flux), method = 'leastsq')
    
    if make_plot:
        bands = ['u', 'g', 'r', 'i', 'z']
        
        plt.figure(figsize = (8,7))
        plt.errorbar(bands, obs_flux, yerr = e_obs_flux, linestyle = 'none', marker = 'None', color = 'k',
                    capsize = 5, label = 'Observed SED')
        plt.plot(bands, get_model_flux(result.params), 'bo', markersize = 10, label = 'Model SED')
        plt.xlabel('Band')
        plt.ylabel('Apparent Flux)')
        #plt.gca().invert_yaxis()
        plt.legend() 
        
    return result

if __name__ == '__main__':
    bsq = BayestarQuery()

    
    
    