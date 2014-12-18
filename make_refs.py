from __future__ import division, print_function
"""
Makes the reference files necessary to do the final drizzling.

This needs to be run from the directory "reduce_acs.py" is in.  Also requires
ureka to be activated.
"""
import os
import shutil
import reduce_acs

ref_scale = 0.03
ref_filter = 'F606W'
refs_dir = 'drizzled_refs'

asnd, _ = reduce_acs.find_target_associations('/Volumes/ErikDroboData/data/ACS/pisc_dwarfs/',
                                              {'HI22': 'PiscA', 'HI23': 'PiscB'})

for i, (objnm, asns) in enumerate(asnd.items()):
    print('Making ref for', objnm, '- object', i+1, 'of', len(asnd))

    refasn = [asn for asn in asns if asn.filter == ref_filter]
    if len(refasn) == 0:
        raise ValueError('Found no association with filter {0} for obj {1}'.format(ref_filter, objnm))
    elif len(refasn) > 1:
        print('Found multiple associations with filter', ref_filter, 'using first one')
    refasn = refasn[:1]

    outnm = objnm + '_ref'
    workingdir = os.path.join(refs_dir, objnm)
    reduce_acs.do_drizzle(refasn, outnm, workingdir, final_scale=ref_scale, final_wcs=True)

    shutil.move(os.path.join(workingdir, outnm) + '_drc_sci.fits', os.path.join(refs_dir, outnm) + '_drc_sci.fits')
    shutil.rmtree(workingdir)
