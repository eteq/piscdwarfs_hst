from __future__ import division

import numpy as np

from scipy import special

from astropy import units as u
from astropy.io import fits
from astropy.constants import h, c, k_B, sigma_sb

FLAMB = u.erg / (u.cm**2 * u.s * u.Angstrom)
TOTPHOTNORM = 30 * special.zeta(3, 1) * sigma_sb / (k_B * np.pi**5)


def normed_planck(lamb, T, fphot=True):
    """
    f_phot planck law, normalized to integrate to unity.
    """
    if lamb.unit.physical_type != 'length':
        raise TypeError('physical_type of lamb is not a length')
    if T.unit.physical_type != 'temperature':
        raise TypeError('physical_type of T is not a temp')

    hnu = h * c / lamb
    kT = k_B * T

    if fphot:
        Bl = (2 * c * lamb**-4) / (np.exp(hnu/kT) - 1)
        tot_phot = TOTPHOTNORM * T**3
        return (Bl / tot_phot).to(1/lamb.unit)
    else:
        Bl = (2 * hnu * c * lamb**-4) / (np.exp(hnu/kT) - 1)
        totE = (sigma_sb * T**4) / np.pi
        return (Bl / totE).to(1/lamb.unit)


def compute_T_ratio_grid(Ts, band1, band2):
    """
    Returns (T, ratio, wleff1, wleff2) (first twosorted on ratio).
    The `band#`s should be (wl, throughput)

    Note that the edge values of ratio will be set to asymptotic numbers
    (so the lowest and highest temps will not match ``Ts``'s)
    """

    wl1, throughput1 = band1
    wl2, throughput2 = band2

    newT = np.sort(Ts)
    newT[-1] *= 1e6  # asymptotic temp, will set to inf later
    newT = newT.reshape(-1, 1)

    int1 = normed_planck(wl1, newT) * throughput1
    int2 = normed_planck(wl2, newT) * throughput2

    dwl1 = np.empty_like(wl1)
    dwl1[:-1] = np.diff(wl1)
    dwl1[-1] = dwl1[-2]
    dwl2 = np.empty_like(wl2)
    dwl2[:-1] = np.diff(wl2)
    dwl2[-1] = dwl2[-2]

    fxperc1 = np.sum(int1 * dwl1, axis=1)
    fxperc2 = np.sum(int2 * dwl2, axis=1)

    leff1 = np.sum(wl1 * throughput1) / np.sum(throughput1)
    leff2 = np.sum(wl2 * throughput2) / np.sum(throughput2)

    ratio = fxperc1 / fxperc2
    # the highest temp is the asymptotic Rayleigh-Jeans value, but set the
    #lowest to the Wien asymptotic value of 0 or inf (depending on which is redder)
    newT[-1] = np.inf

    if leff1 < leff2:
        ratio[0] = 0
    else:
        ratio[0] = np.inf

    #resort on ratio
    sorti = np.argsort(ratio)
    return newT[sorti].ravel(), ratio[sorti], leff1, leff2


def eye_response(Ts, fn='smj10q.csv', normedat=None):
    """
    Response/colors the human eye gets from a BB of the specified temp

    ``normedat`` != None means a temperature to normalize everythin to 1 at
    """
    Tshape = Ts.shape

    # load everything
    wl_cone, logthroughput_L, logthroughput_M, logthroughput_S = np.loadtxt(fn, delimiter=',').T
    wl_cone = wl_cone*u.nm
    throughput_L = 10**logthroughput_L
    throughput_M = 10**logthroughput_M
    throughput_S = 10**logthroughput_S

    # compute the integrands
    intL = normed_planck(wl_cone, Ts.reshape(-1, 1))*throughput_L
    intM = normed_planck(wl_cone, Ts.reshape(-1, 1))*throughput_M
    intS = normed_planck(wl_cone, Ts.reshape(-1, 1))*throughput_S

    #do the integral
    dwl_cone = np.diff(wl_cone).mean()
    respS = np.sum(intS*dwl_cone, axis=1)
    respM = np.sum(intM*dwl_cone, axis=1)
    respL = np.sum(intL*dwl_cone, axis=1)

    if normedat:
        if normedat.shape != ():
            raise ValueError('normedat must be a scalar temperature or None')
        nresps = eye_response(normedat.ravel(), fn, None)
        respS /= nresps[0]
        respM /= nresps[1]
        respL /= nresps[2]

    return respS.reshape(Tshape), respM.reshape(Tshape), respL.reshape(Tshape)


def T_to_rgb(Ts, fn='smj10q.csv', normedat=None):
    """
    Uses `eye_response` to determine RGB colors.  Note that these are normalized
    so that the *maximum* of the 3 cones sets what 1 is, so saturated blue is
    impossible

    Returned array is on [0,1] and has shape (T_shape..., 3)
    """
    respS, respM, respL = eye_response(Ts, fn, normedat)
    imarr = np.array([respL.value, respM.value, respS.value])
    imarr = imarr / np.max(imarr, axis=0)
    return imarr.transpose(1, 2, 0)


def two_color_image(fn1, fn2, rng1, rng2, savefn=None, sl1=None, sl2=None):
    """
    This takes two file names, loads them, remaps the values into the
    bounds specified by `rng1` and `rng2`, and combines them as R/B channels,
    with G the average of the two.

    fn1 should be bluer, fn2 should be redder
    """
    import PIL.Image

    d1 = fits.getdata(fn1)
    if sl1 is not None:
        d1 = d1[sl1]

    d2 = fits.getdata(fn2)

    if sl2 is not None:
        d2 = d2[sl2]

    l1 = min(rng1)
    u1 = max(rng1)
    l2 = min(rng2)
    u2 = max(rng2)

    dscale1 = (d1 - l1)/(u1 - l1)
    dscale2 = (d2 - l2)/(u2 - l2)

    dscale1[dscale1 > 1] = 1
    dscale1[dscale1 < 0] = 0
    dscale2[dscale2 > 1] = 1
    dscale2[dscale2 < 0] = 0

    imarr = np.array([dscale1, (dscale1 + dscale2) / 2, dscale2]).transpose(1, 2, 0)
    imarr = (255*imarr).astype('uint8')

    im = PIL.Image.fromarray(imarr, mode='RGB')
    if savefn:
        im.save(savefn)
    return im
