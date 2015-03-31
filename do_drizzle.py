from __future__ import division, print_function
"""
Makes does the drizzling for the actual science files

This needs to be run from the directory "reduce_acs.py" is in.  Also requires
ureka to be activated.
"""
import os
import reduce_acs

# the jref seems to *sometimes* be needed... not clear on when exactly or why
os.environ['jref'] = '/Users/erik/astrodata/HST_jref'

refs_dir = 'drizzled_refs'
pixfrac = 0.6
build = False
dirprefix = 'drizzled_build_' if build else 'drizzled_'
final_wht_type = None

asnd, _ = reduce_acs.find_target_associations('raw', {'HI22': 'PiscA', 'HI23': 'PiscB'})

for objnm, asns in asnd.items():
    reffn = os.path.abspath(os.path.join(refs_dir, objnm + '_ref_drc_sci.fits'))
    for i, asn in enumerate(asns):
        print('Starting', i+1, 'of', len(asns), 'associations for ', objnm)
        if final_wht_type is None:
            whtending = ''
        else:
            whtending = '_' + final_wht_type
        reduce_acs.do_drizzle([asn], objnm + '_' + asn.filter + whtending, dirprefix,
                              final_refimage=reffn, final_wcs=True,
                              final_pixfrac=pixfrac, final_wht_type=final_wht_type,
                              build=build)

    print('Doing combination of all associations for', objnm)
    reduce_acs.do_drizzle(asns, objnm + '_allfilters_ivm', dirprefix,
                          final_refimage=reffn, final_wcs=True,
                          final_pixfrac=pixfrac, final_wht_type='IVM',
                          build=build)
    reduce_acs.do_drizzle(asns, objnm + '_allfilters_err', dirprefix,
                          final_refimage=reffn, final_wcs=True,
                          final_pixfrac=pixfrac, final_wht_type='ERR',
                          build=build)
