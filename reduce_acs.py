from __future__ import division, print_function

import os
import shutil

import numpy as np

from astropy.io import fits
from astropy import units as u


class Association(object):
    def __init__(self, asnfn):
        self.datadir = os.path.abspath(os.path.split(asnfn)[0])
        asn = fits.open(asnfn)

        self.asn_header = asn[0].header
        self.table = asn[1].data

        self._populate_asn_attributes()
        self._set_flt_info()

    def _populate_asn_attributes(self):
        """
        Use the header and table to figure out what
        """
        from astropy.coordinates import SkyCoord

        hdr = self.asn_header

        self.target_name = hdr['TARGNAME']
        self.target_coords = SkyCoord(hdr['RA_TARG']*u.deg, hdr['RA_TARG']*u.deg)
        self.instrument = hdr['INSTRUME']
        self.detector = hdr['DETECTOR']
        self.propid = hdr['PROPOSID']
        self.data = hdr['DATE']

        self.exposure_names = []
        self.product_name = None

        for nm, typ, cal in self.table:
            if not cal:
                raise ValueError('File {0} was not calibrated!'.format(nm))
            if typ == 'EXP-DTH':
                self.exposure_names.append(nm.lower())
            elif typ == 'PROD-DTH':
                if self.product_name:
                    raise ValueError('Found *two* products: "{0}" and '
                                     '"{1}"'.format(self.product_name, nm))
                self.product_name = nm.lower()
            else:
                raise ValueError('Unrecognized type "{0}" for file {1}'.format(typ, nm))


    def _set_flt_info(self):
        """
        Determines extra info from the flt of the first exposure
        """
        fltfn = self.flts[0]
        self.flt_header = hdr = fits.getheader(fltfn, 0)
        self.filter = (hdr['FILTER1'], hdr['FILTER2'])

    @property
    def flts(self):
        return [os.path.join(self.datadir, basefn + '_flt.fits') for basefn in self.exposure_names]

    @property
    def flcs(self):
        return [os.path.join(self.datadir, basefn + '_flc.fits') for basefn in self.exposure_names]

    @property
    def drz(self):
        return os.path.join(self.datadir, self.product_name + '_drz.fits')

    @property
    def drc(self):
        return os.path.join(self.datadir, self.product_name + '_drc.fits')


def find_target_associations(datadir, targrenames=None):
    from glob import glob

    asnfns = glob(os.path.join(datadir, '*asn*.fits'))
    asns = [Association(fn) for  fn in asnfns]
    targdct = dict()
    for asn in asns:
        targ_asns = targdct.setdefault(asn.target_name, [])
        targ_asns.append(asn)

    if targrenames:
        for old, new in targrenames.items():
            targdct[new] = targdct.pop(old)

    return targdct, asns

default_dolphot_params = {
'img_apsky': '15 25',
'UseWCS': '1',
'RAper': '4',
'RChi': '2.0',
'RSky0': '15',
'RSky1': '35',
'SkipSky': '2',
'SkySig': '2.25',
'SecondPass': '5',
'SigFindMult': '0.85',
'MaxIT': '25',
'NoiseMult': '0.10',
'FSat': '0.999',
'ApCor': '1',
'RCentroid': '2',
'PosStep': '0.25',
'dPosMax': '2.5',
'RCombine': '1.5',
'RPSF': '10',
'SigPSF': '5.0',
'PSFres': '1',
'PSFPhot': '1',
'FitSky': '1',
'Force1': '0',
'Align': '2',
'Rotate': '1',
'ACSuseCTE': '1',
'FlagMask': '4',
'ACSpsfType': '0'
}

def do_dolphot(asns, dest_dir, allowexistingdata=False, dolphotpath=None, cte=True):
    from warnings import warn
    from dolphot_runner import DolphotRunner


    #first check that the destination dir exists and copy over the data
    if os.path.exists(dest_dir):
        if allowexistingdata == 'clobber':
            print("Deleting directory", dest_dir)
            shutil.rmtree(dest_dir)
        elif allowexistingdata:
            warn('Destination directory "{0}" already exists.'.format(dest_dir))
        else:
            raise IOError('Destination directory "{0}" already exists!'.format(dest_dir))
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    rawfiles = []
    rawexposures = []
    for asn in asns:
        rawfiles.append(asn.drc if cte else asn.drz)
        rawfiles.extend(asn.flcs if cte else asn.flts)
        rawexposures.extend(asn.flcs if cte else asn.flts)
        for fn in rawfiles:
            targfn = os.path.join(dest_dir, os.path.split(fn)[-1])
            if os.path.exists(targfn):
                print(targfn, 'already exists, not copying')
            else:
                print('Copying', fn, '->', targfn)
                shutil.copy(fn, targfn)


    #replace the 'acsmask' command with various acs commands later
    acs_runner = DolphotRunner('acsmask', workingdir=dest_dir, execpathordirs=dolphotpath,
                                          paramfile=None, logfile=None)
    dolphot_runner = DolphotRunner('dolphot', workingdir=dest_dir, execpathordirs=dolphotpath,
                                              params=default_dolphot_params)

    print('\n...Running acsmask...\n')
    acs_runner(*rawexposures)

    print('\n...Running splitgroups...\n')
    acs_runner.cmd = 'splitgroups'
    #return acs_runner(*rawfiles)


