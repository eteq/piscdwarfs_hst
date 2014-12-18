from __future__ import division, print_function
"""
Makes does the drizzling for the actual science files

This needs to be run from the directory "reduce_acs.py" is in.  Also requires
ureka to be activated.

Also, note that
"""
import os
import reduce_acs

refs_dir = 'drizzled_refs'
pixfrac = 0.6
# the jref seems to *sometimes* be needed... not clear on when exactly or why
os.environ['jref'] = '/Users/erik/astrodata/HST_jref'

asnd, _ = reduce_acs.find_target_associations('/Volumes/ErikDroboData/data/ACS/pisc_dwarfs/',
                                              {'HI22': 'PiscA', 'HI23': 'PiscB'})

for objnm, asns in asnd.items():
    reffn = os.path.abspath(os.path.join(refs_dir, objnm + '_ref_drc_sci.fits'))
    for i, asn in enumerate(asns):
        print('Starting', i+1, 'of', len(asns), 'associations for ', objnm)
        reduce_acs.do_drizzle([asn], objnm + '_' + asn.filter, 'drizzled_',
                              final_refimage=reffn, final_wcs=True, final_pixfrac=pixfrac)

    print('Doing combination of all associations for', objnm)
    reduce_acs.do_drizzle(asns, objnm + '_allfilters_ivm', 'drizzled_',
                          final_refimage=reffn, final_wcs=True,
                          final_pixfrac=pixfrac, final_wht_type='IVM')
    reduce_acs.do_drizzle(asns, objnm + '_allfilters_err', 'drizzled_',
                          final_refimage=reffn, final_wcs=True,
                          final_pixfrac=pixfrac, final_wht_type='ERR')
