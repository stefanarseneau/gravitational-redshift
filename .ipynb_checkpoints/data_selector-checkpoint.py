# Install a pip package in the current Jupyter kernel
import sys

### General
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.io import fits
from astropy.table import Table, vstack, join
from tqdm import tqdm
import pandas as pd
import os

### Query
from astroquery.sdss import SDSS
from astroquery.gaia import Gaia

def search(bestObjID):
    sourceobjid = []
    plate = []
    mjd = []
    fiberid = []
    sclass = []
    u = []
    g = []
    r = []
    i = []
    z = []
    
    url = []
    
    drops = []
    
    for i in tqdm(range(len(bestObjID))):
        notfound = False
        j = np.where(( bestObjID[i] == wide_convert['bestObjID'] ))
        
        try:
            k = j[0][0]
        except:
            notfound = True
            
        if not notfound: 
            try:
                sourceobjid.append(wide_convert['specObjID'][k])
                plate.append(wide_convert['plate'][k])
                mjd.append(wide_convert['mjd'][k])
                fiberid.append(wide_convert['fiberID'][k])
                try:
                    sclass.append(wide_convert['subclass'][k])
                except:
                    pass
                u.append(wide_convert['u'][k])
                g.append(wide_convert['g'][k])
                r.append(wide_convert['r'][k])
                i.append(wide_convert['i'][k])
                z.append(wide_convert['z'][k])
            except:
                notfound = True
        if notfound:
            drops.append(i)  
            
    for i in tqdm(range(len(sourceobjid))):
        xid = SDSS.get_spectra_async(plate=plate[i], fiberID=fiberid[i], mjd=mjd[i])
        url.append(str(xid[0]).split(' ')[4])
    
    
    return drops, sourceobjid, plate, mjd, fiberid, url, u, g, r,i,z, sclass

def select_sdss4(el_badry = '../external-dat/binaries/all_columns_catalog.fits', outfile = 'data/catalog_sdss4.csv'):
    
    hdul = Table.read(el_badry, format='fits')
    df = hdul.to_pandas()
    
    print('Loaded El-Badry Catalog')
    
    catalog = df[ df['binary_type'] == 'WDMS']
    catalog.reset_index(inplace=True, drop=True)
    
    print('Querying Gaia...')
    
    ADQL_CODE1 = "SELECT \
        sdss.original_ext_source_id as bestobjid,\
        gaia_source.source_id\
        FROM gaiaedr3.gaia_source \
        JOIN gaiaedr3.sdssdr13_best_neighbour as sdss\
        ON gaia_source.source_id = sdss.source_id      \
        WHERE gaia_source.source_id IN {}\
    ".format(tuple(catalog['source_id1']))
    ADQL_CODE2 = "SELECT \
        sdss.original_ext_source_id as bestobjid,\
        gaia_source.source_id\
        FROM gaiaedr3.gaia_source \
        JOIN gaiaedr3.sdssdr13_best_neighbour as sdss\
        ON gaia_source.source_id = sdss.source_id      \
        WHERE gaia_source.source_id IN {}\
    ".format(tuple(catalog['source_id2']))
    
    job1 = Gaia.launch_job(ADQL_CODE1,dump_to_file=False)
    job2 = Gaia.launch_job(ADQL_CODE2,dump_to_file=False)
    
    d1 = job1.get_results()
    d2 = job2.get_results()
    
    print('Done!')
          
    drops = []
    bestobjid1 = []
    bestobjid2 = []
    
    for i in tqdm (range(len(catalog))):
        notfound = False
        a = np.where(d1['source_id'] == catalog['source_id1'][i])
        b = np.where(d2['source_id'] == catalog['source_id2'][i])
        
        try:
            j = a[0][0]
            k = b[0][0]
        except:
            notfound = True
            
        if not notfound: 
            try:
                bestobjid1.append(d1['bestobjid'][j])
                bestobjid2.append(d2['bestobjid'][k])
            except:
                notfound = True
        if notfound:
            drops.append(i)    
            
    catalog = catalog.drop(drops)
    catalog['bestobjid1'] = bestobjid1
    catalog['bestobjid2'] = bestobjid2
    catalog.reset_index(inplace=True, drop=True)
    tcatalog1 = catalog
          
    print('Querying SDSS...')
    
    SDSS_CODE1 = """select sp.bestObjID, sp.specObjID, sp.plate, sp.fiberID, sp.mjd, sp.subclass, ph.u, ph.g, ph.r, ph.i, ph.z
        from dbo.SpecObjAll as sp
        join dbo.PhotoObjAll as ph 
        on sp.bestObjID = ph.objID
        where sp.bestObjID > 1237648702985666868
        and sp.bestObjID < 1237660529738105365
        and sp.class = 'STAR'"""
    SDSS_CODE2 = """select sp.bestObjID, sp.specObjID, sp.plate, sp.fiberID, sp.mjd, sp.subclass, ph.u, ph.g, ph.r, ph.i, ph.z
        from dbo.SpecObjAll as sp
        join dbo.PhotoObjAll as ph 
        on ph.objID = sp.bestObjID
        where sp.bestObjID > 1237660529738105365
        and sp.bestObjID < 1237670529738105366
        and sp.class = 'STAR'"""
    SDSS_CODE3 = """select sp.bestObjID, sp.specObjID, sp.plate, sp.fiberID, sp.mjd, sp.subclass, ph.u, ph.g, ph.r, ph.i, ph.z
        from dbo.SpecObjAll as sp
        join dbo.PhotoObjAll as ph 
        on ph.objID = sp.bestObjID
        where sp.bestObjID > 1237670529738105366
        and sp.bestObjID < 1237680529738105576
        and sp.class = 'STAR'"""
    
    convert1= SDSS.query_sql(SDSS_CODE1)
    convert2= SDSS.query_sql(SDSS_CODE2)
    convert3= SDSS.query_sql(SDSS_CODE3)
    
    twide_convert = vstack([convert1, convert2, convert3])
    wide_convert = twide_convert
          
    print('Done!')
    
    drops1, sourceobjid1, plate1, mjd1, fiberID1, url1 ,u1,g1,r1,i1,z1, subclass1= search( catalog['bestobjid1'] )
    
    catalog = catalog.drop(drops1)
    catalog['specobjid1'] = sourceobjid1
    catalog['plate1'] = plate1
    catalog['mjd1'] = mjd1
    catalog['fiberID1'] = fiberID1
    catalog['url1'] = url1
    catalog['subclass1'] = subclass1
    
    catalog['u1'] = u1
    catalog['g1'] = g1
    catalog['r1'] = r1
    catalog['i1'] = i1
    catalog['z1'] = z1
    catalog.reset_index(inplace=True, drop=True)
    
    drops2, sourceobjid2, plate2, mjd2, fiberID2, url2, u2,g2,r2,i2,z2, subclass2 = search( catalog['bestobjid2'] )
    
    catalog = catalog.drop(drops2)
    catalog['specobjid2'] = sourceobjid2
    catalog['plate2'] = plate2
    catalog['mjd2'] = mjd2
    catalog['fiberID2'] = fiberID2
    catalog['url2'] = url2
    catalog['subclass2'] = subclass2
    
    catalog['u2'] = u2
    catalog['g2'] = g2
    catalog['r2'] = r2
    catalog['i2'] = i2
    catalog['z2'] = z2
    catalog.reset_index(inplace=True, drop=True)
          
    print('Saving catalog to {}'.format(outfile))
    
    catalog.to_csv(outfile)
    
def get_catalog(path, build_spectra = True):
    catalog = pd.read_csv(path)
    
    if build_spectra:
        drops = []

        flux1 = []
        wavelength1 = []
        ivar1 = []
        
        flux2 = []
        wavelength2 = []
        ivar2 = []
        
        for i in tqdm( range(len(catalog))):
            try:
                spec1 = fits.open(catalog['url1'][i], allow_insecure=True)
                spec2 = fits.open(catalog['url2'][i], allow_insecure=True)
                
                flux1.append(spec1[1].data['flux'])
                wavelength1.append(10**spec1[1].data['loglam'])
                ivar1.append(spec1[1].data['ivar'])
                
                flux2.append(spec2[1].data['flux'])
                wavelength2.append(10**spec2[1].data['loglam'])
                ivar2.append(spec2[1].data['ivar'])
                
            except:
                drops.append(i)
                                
        catalog = catalog.drop(drops)
        catalog['flux1'] = flux1
        catalog['wavelength1'] = wavelength1
        catalog['ivar1'] = ivar1
        catalog['flux2'] = flux2
        catalog['wavelength2'] = wavelength2
        catalog['ivar2'] = ivar2
        catalog.reset_index(inplace=True, drop=True)
        
    return catalog

def plot_binary_spectra(catalog, lines, survey):    
    for i in tqdm(range(len(catalog))):
        plt.figure(figsize=(20,16))
        
        plt.subplot(211)
        plt.plot(catalog['wavelength1'][i], catalog['flux1'][i])
        plt.ylim(bottom = -1.5)
        bottom, top = plt.ylim()
        plt.vlines(x=lines, ymin=bottom, ymax = top, colors='black', ls=':', lw=2, label='vline_single - full height')
        plt.grid()
        plt.ylabel(r'Flux [$10^{-17}$ ergs/s/cm2/A]')
        plt.xlabel(r'Wavelength [A]')
        plt.title('Object 1 Spectrum ({})'.format(i))
        #ax = plt.gca()
        
        plt.subplot(212)
        plt.plot(catalog['wavelength2'][i], catalog['flux2'][i])
        plt.ylim(bottom = -1.5)
        
        plt.vlines(x=lines, ymin=bottom, ymax = top, colors='black', ls=':', lw=2, label='vline_single - full height')
        plt.grid()
        plt.ylabel(r'Flux [$10^{-17}$ ergs/s/cm2/A]')
        plt.xlabel(r'Wavelength [A]')
        plt.title('Object 2 Spectrum ({})'.format(i))
        #ax = plt.gca()
        
        plt.ioff()
        plt.savefig('spectra/{}/binary{}.jpg'.format(survey, i))
        

            