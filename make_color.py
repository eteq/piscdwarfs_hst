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


def compute_T_ratio_grid(Ts, band1, band2, mags=False):
    """
    Returns (T, ratio, wleff1, wleff2) (first twosorted on ratio).
    The `band#`s should be (wl, throughput)

    The ratio is in the sense of flux_b1/flux_b2

    Note that the edge values of ratio will be set to asymptotic numbers
    (so the lowest and highest temps will not match ``Ts``'s)

    `band1` and `band2` should be (2, N) of wl (in Angstroms) and flux

    If `mags` is True, `ratio` is a magnitude difference (color) rather than a
    flux ratio
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

    if mags:
        ratio = -2.5*np.log10(ratio)

    return newT[sorti].ravel(), ratio[sorti], leff1, leff2


def band_ratio_plot(Ts, band1, band2, colormag=False, normedat=5800*u.K, logy=True):
    from matplotlib import pyplot as plt

    Tgrid, ratio, leff1, leff2 = compute_T_ratio_grid(Ts, band1, band2, colormag)
    plt.gca().set_axis_bgcolor('k')
    if logy:
        Tgridy = np.log10(Tgrid/u.K)
    else:
        Tgridy = Tgrid
    plt.scatter(ratio, Tgridy, lw=0, c=T_to_rgb(Tgrid, normedat=normedat))
    if normedat:
        plt.axhline(np.log10(normedat/u.K).value if logy else normedat.to(u.K), c='w')

    if colormag:
        plt.xlabel('color')
    else:
        plt.xlabel('flux ratio')

    if logy:
        plt.ylabel('log T [K]')
    else:
        plt.ylabel('T [K]')
        plt.ylim(Ts.max().value/-100, Ts.max().value)


def eye_response(Ts, fn='smj10q.csv', normedat=None):
    """
    Response/colors the human eye gets from a BB of the specified temp

    ``normedat`` != None means a temperature to normalize everythin to 1 at
    """
    if not hasattr(Ts, 'unit') or Ts.unit.physical_type != 'temperature':
        raise TypeError('input to eye_response must be a temp quantity')
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


def T_to_rgb(Ts, fn='smj10q.csv', normedat=None, modulatewith=None):
    """
    Uses `eye_response` to determine RGB colors.  Note that these are normalized
    so that the *maximum* of the 3 cones sets what 1 is, so saturated blue is
    impossible

    Returned array is on [0,1] or is multiplied by `modulatewith` and has shape (T_shape..., 3)
    """
    respS, respM, respL = eye_response(Ts, fn, normedat)

    imarr = np.array([respL.value, respM.value, respS.value])
    imarr = imarr / np.max(imarr, axis=0)

    if modulatewith is not None:
        imarr = imarr * modulatewith


    #transpose only the first axis to the end
    totrans = range(1, len(imarr.shape))
    totrans.append(0)
    return imarr.transpose(totrans)


def rescale(img, lower, upper, postfunc=None, lowerfill=None, upperfill=None):
    rimg = (img-lower)/(upper-lower)
    if lowerfill is not None:
        lowermsk = rimg <= 0
        rimg[lowermsk] = lowerfill
    if upperfill is not None:
        uppermsk = rimg >= 1
        rimg[uppermsk] = upperfill

    if postfunc:
        return postfunc(rimg)
    else:
        return rimg


def modulate_color_image(colorimg, lumimg):
    totrans = range(len(colorimg.shape))
    totrans.insert(0, totrans.pop(-1))
    totrans2 = range(1, len(colorimg.shape))
    totrans2.append(0)

    return (colorimg.transpose(totrans) * lumimg).transpose(totrans2)


def floatarr_to_image(arr, fn, notfinvalue=0):
    """
    arr should be (x, y, 3 or 4)
    Will be clamped so that [0,1] -> [0,255]
    """
    import PIL.Image

    imarr = arr.copy()

    imarr[~np.isfinite(imarr)] = notfinvalue
    imarr[imarr>1] = 1
    imarr[imarr<0] = 0

    imarr = (255*imarr).astype('uint8')

    im = PIL.Image.fromarray(imarr, mode='RGB')
    if fn is not None:
        im.save(fn)
    return im


def prettify_acs_image(img1fn, img2fn, imgallfn, outfn,
                       rescalelower=0., rescaleupper=.15, rescalefunc=None,
                       msk=None, normedat=5800*u.K,
                       band1fn='wfc_F606W.dat', band2fn='wfc_F814W.dat', convsize=1,
                       Tsforgrid=np.logspace(2.75, 5, 250)*u.K,
                       mintemp=2500*u.K,
                       finalsmoothkernel=None):
    from astropy import convolution

    print('Loading data')
    band1 = np.loadtxt(band1fn)
    band1 = (band1[:, 0]*u.angstrom, band1[:, 1])
    band2 = np.loadtxt(band2fn)
    band2 = (band2[:, 0]*u.angstrom, band2[:, 1])

    Tratios, ratios, leff1, leff2 = compute_T_ratio_grid(Tsforgrid, band1, band2)

    img1 = fits.getdata(img1fn)
    img2 = fits.getdata(img2fn)
    imgall = fits.getdata(imgallfn)

    if msk is not None:
        print('Masking data')
        img1 = img1[msk]
        img2 = img2[msk]
        imgall = imgall[msk]

    print("Computing flux ratio")
    ratioimg = img1/img2
    print('Interpolating ratios to temperatures')
    Timg = np.interp(ratioimg, ratios, Tratios, left=-1,right=-2)*u.K

    print('Converting to RGB')
    colorimg = T_to_rgb(Timg,normedat=normedat)
    rescaledall = rescale(imgall, rescalelower, rescaleupper, rescalefunc, 0, 1)

    #convolve and eliminate bad temps.  Control definition of bad through the t grid
    if convsize:
        print('Convolving and cleaning')
        msk = ((Timg<mintemp)|~np.isfinite(Timg.value))
        fixedrescaledall = rescaledall.copy()
        fixedrescaledall[msk] = 0
        k = convolution.Gaussian2DKernel(convsize)
        k.array[k.shape[0]//2,k.shape[0]//2] = 0
        convimg = convolution.convolve(fixedrescaledall,convolution.Gaussian2DKernel(1), normalize_kernel=True)
        fixedrescaledall[msk] = convimg[msk]
    else:
        fixedrescaledall = rescaledall

    outimgarr = modulate_color_image(colorimg, fixedrescaledall)

    if finalsmoothkernel:
        print('Doing final smoothing')
        outimgarr = [convolution.convolve(outimgarr[:, :, i], finalsmoothkernel) for i in range(3)]
        outimgarr = np.array(outimgarr, copy=False).transpose(2, 0, 1)

    print('Saving to', outfn)
    floatarr_to_image(outimgarr, outfn)

    return locals()



def simple_two_color_image(fn1, fn2, rng1, rng2, savefn=None, sl1=None, sl2=None):
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
