from __future__ import division, print_function

import numpy as np
from scipy.special import erf, erfc

import emcee
from astropy import units as u

MINF = -np.inf


class RGBModel(object):
    PARAM_NAMES = 'tipmag, alphargb, alphaother, fracother'.split(', ')

    def __init__(self, magdata, magunc=None, tipprior=None):

        self.magdata = magdata
        self.magunc = magunc

        self.maxdata = np.max(magdata)
        self.mindata = np.min(magdata)

        self.tipprior = tipprior

    def lnpri(self, tipmag, alphargb, alphaother, fracother):
        #flat priors on everything to start with
        lpri = 0.0

        if not (0 < fracother < 1):
            return MINF
        if not (-5 < alphargb < 5):
            return MINF
        if not (-5 < alphaother < 5):
            return MINF
        if not (self.mindata < tipmag < self.maxdata):
            return MINF

        if self.tipprior is not None:
            if not (self.tipprior[0] < tipmag < self.tipprior[1]):
                return MINF

        return lpri

    def lnprob(self, tipmag, alphargb, alphaother, fracother):
        """
        This does *not* sum up the lnprobs - that goes in __call__.  Instead it
        gives the lnprob per data point
        """
        dmags = self.magdata - tipmag
        if self.magunc is None:
            rgbmsk = dmags > 0
            lnpall = np.zeros_like(dmags)

            lnpall[rgbmsk] = alphargb * dmags[rgbmsk]
            lnpall[~rgbmsk] = alphaother * dmags[~rgbmsk] + np.log(fracother)

            eterm1 = 1 - np.exp(alphaother*(self.mindata - tipmag))
            eterm2 = np.exp(alphargb*(self.maxdata - tipmag)) - 1
            lnN = np.log(fracother * eterm1 / alphaother + eterm2 / alphargb)

            return lnpall - lnN
        else:
            return np.log(exp_gauss_conv(dmags, self.alphargb, self.alphaother,
                                                self.fracother, self.magunc))

    def __call__(self, params):
        lpri = self.lnpri(*params)
        if lpri == MINF:
            return MINF
        else:
            return lpri + np.sum(self.lnprob(*params))

    def make_and_run_sampler(self, nwalkers=len(PARAM_NAMES)*6, niters=0, burnin=None, tipmag0=23):
        sampler = emcee.EnsembleSampler(nwalkers, len(self.PARAM_NAMES), self)
        if niters > 0:
            p0 = self.initalize_params(nwalkers, tipmag0=tipmag0)
            if burnin:
                resb = sampler.run_mcmc(p0, burnin)
                sampler.reset()
                res = sampler.run_mcmc(resb[0], niters)
            else:
                res = sampler.run_mcmc(p0, niters)
        else:
            res = None
        return sampler, res

    def initalize_params(self, nwalkers, tipmag0=23, alphargb0=1.5, alphaother0=0.5, fracother0=.1):
        return emcee.utils.sample_ball([tipmag0, alphargb0, alphaother0, fracother0],
                                       [1e-3]*len(RGBModel.PARAM_NAMES), nwalkers)

    def plot_lnprob(self, tipmag, alphargb, alphaother, fracother, magrng=100, doplot=True, delog=False):
        """
        Plots (optionally) and returns arrays suitable for plotting the pdf. If
        `magrng` is a scalar, it gives the number of samples over the data
        domain.  If an array, it's used as the x axis.
        """
        from astropy.utils import isiterable
        from matplotlib import pyplot as plt



        if isiterable(magrng):
            fakemod = self.__class__(magrng)
        else:
            fakemod = self.__class__(np.linspace(self.mindata, self.maxdata, magrng))

        lnpb = fakemod.lnprob(tipmag, alphargb, alphaother, fracother)
        if delog:
            lnpb = np.exp(lnpb - np.min(lnpb))

        if doplot:
            plt.plot(fakemod.magdata, lnpb)

        return fakemod.magdata, lnpb


def exp_gauss_conv(x, a=1, b=1, F=0, s=.1):
    """
    Convolution of broken power law w/ gaussian.
    """
    A = np.exp(a*x+a**2*s**2/2.)
    B = np.exp(b*x+b**2*s**2/2.)
    ua = (x+a*s**2)*2**-0.5/s
    ub = (x+b*s**2)*2**-0.5/s
    return (A*(1+erf(ua))+F*B*erfc(ub))
