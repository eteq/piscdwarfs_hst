from __future__ import division, print_function

import numpy as np
from scipy.special import erf, erfc

import emceemr
from astropy import units as u

MINF = -np.inf


class RGBModel(emceemr.Model):
    param_names = 'tipmag, alphargb, alphaother, fracother'.split(', ')

    def __init__(self, magdata, magunc=None, priors=None,
                       uncfunc=None, biasfunc=None, complfunc=None):

        self.magdata = np.array(magdata)

        self.maxdata = np.max(magdata)
        self.mindata = np.min(magdata)

        self._magunc = magunc
        self._uncfunc = uncfunc
        self._biasfunc = biasfunc
        self._complfunc = complfunc
        self._validate_lnprob_func()

        super(RGBModel, self).__init__(priors)

    def _validate_lnprob_func(self):
        """
        Checks that the various ways of giving uncertainties or not make sense
        """
        if self.magunc is None:
            if self.uncfunc is not None:
                self._lnprob_func = _lnprob_uncfuncs
            else:
                self._lnprob_func = _lnprob_no_unc
        elif self.uncfunc is not None:
            raise ValueError('Cannot give both uncertainties and the various uncfuncs')
        else:
            self._lnprob_func = _lnprob_w_unc


    def lnprob(self, tipmag, alphargb, alphaother, fracother):
        """
        This does *not* sum up the lnprobs - that goes in __call__.  Instead it
        gives the lnprob per data point
        """
        self._lnprob_func(tipmag, alphargb, alphaother, fracother)


    def _lnprob_no_unc(self, tipmag, alphargb, alphaother, fracother):
        dmags = self.magdata - tipmag
        rgbmsk = dmags > 0
        lnpall = np.zeros_like(dmags)

        lnpall[rgbmsk] = alphargb * dmags[rgbmsk]
        lnpall[~rgbmsk] = alphaother * dmags[~rgbmsk] + np.log(fracother)

        eterm1 = 1 - np.exp(alphaother*(self.mindata - tipmag))
        eterm2 = np.exp(alphargb*(self.maxdata - tipmag)) - 1
        lnN = np.log(fracother * eterm1 / alphaother + eterm2 / alphargb)

        return lnpall - lnN

    def _lnprob_w_unc(self, tipmag, alphargb, alphaother, fracother):
        dmag_upper = self.maxdata - tipmag
        dmag_lower = self.mindata - tipmag
        return np.log(self._exp_gauss_conv_normed(self.magdata - tipmag,
                                                  alphargb, alphaother,
                                                  fracother, self.magunc,
                                                  dmag_lower, dmag_upper))

    def _lnprob_uncfuncs(self, tipmag, alphargb, alphaother, fracother):
        raise NotImplementedError

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

    @staticmethod
    def _exp_gauss_conv_normed(x, a, b, F, s, x_lower, x_upper):
        # from scipy.integrate import quad
        # N = quad(exp_gauss_conv, x_lower, x_upper, args=(a, b, F, np.mean(s)))[0]
        # return exp_gauss_conv(x, a, b, F, s)/N
        norm_term_a = RGBModel._exp_gauss_conv_int(x_upper, a, s, g=1) - RGBModel._exp_gauss_conv_int(x_lower, a, s, g=1)
        norm_term_b = RGBModel._exp_gauss_conv_int(x_upper, b, s, g=-1) - RGBModel._exp_gauss_conv_int(x_lower, b, s, g=-1)
        return RGBModel._exp_gauss_conv(x, a, b, F, s)/(norm_term_a + F * norm_term_b)

    @staticmethod
    def _exp_gauss_conv(x, a, b, F, s):
        """
        Convolution of broken power law w/ gaussian.
        """
        A = np.exp(a*x+a**2*s**2/2.)
        B = np.exp(b*x+b**2*s**2/2.)
        ua = (x+a*s**2)*2**-0.5/s
        ub = (x+b*s**2)*2**-0.5/s
        return (A*(1+erf(ua))+F*B*erfc(ub))

    @staticmethod
    def _exp_gauss_conv_int(x, ab, s, g=1):
        """
        Integral for a *single* term of exp_gauss_conv.
        g should be 1/-1
        """
        prefactor = np.exp(-ab**2*s**2 / 2.) / ab
        term1 = np.exp(ab*(ab*s**2 + x))*(1 + g * erf((ab*s**2 + x)*2**-0.5/s))
        term2 = np.exp(ab**2*s**2 / 2.)*g*erf(x * 2**-0.5 / s)
        return prefactor*(term1 - term2)

    #properties for the alternate uncertainty functions
    @property
    def magunc(self):
        return self._magunc
    @magunc.setter
    def magunc(self, value):
        oldval = self._magunc
        self._magunc = value
        try:
            self._validate_lnprob_func()
        except:
            self._magunc = oldval
            raise
    @property
    def uncfunc(self):
        return self._uncfunc
    @uncfunc.setter
    def uncfunc(self, value):
        oldval = self._uncfunc
        self._uncfunc = value
        try:
            self._validate_lnprob_func()
        except:
            self._uncfunc = oldval
            raise
    @property
    def biasfunc(self):
        return self._biasfunc
    @biasfunc.setter
    def biasfunc(self, value):
        oldval = self._biasfunc
        self._biasfunc = value
        try:
            self._validate_lnprob_func()
        except:
            self._biasfunc = oldval
            raise
    @property
    def complfunc(self):
        return self._complfunc
    @complfunc.setter
    def complfunc(self, value):
        oldval = self._complfunc
        self._complfunc = value
        try:
            self._validate_lnprob_func()
        except:
            self._complfunc = oldval
            raise


class NormalColorModel(emceemr.Model):
    param_names = 'colorcen, colorsig, askew'.split(', ')
    has_blobs = True

    def __init__(self, magdata, tipdistr, colordata, colorunc, nstarsbelow=100,
                       priors=None):

        self.magdata = np.array(magdata)
        self.colordata = np.array(colordata)
        self.colorunc = None if colorunc is None else np.array(colorunc)

        self.tipdistr = np.array(tipdistr)
        self._len_tipdistr = self.tipdistr.size
        self.nstarsbelow = nstarsbelow

        super(NormalColorModel, self).__init__(priors)

    def lnprob(self, colorcen, colorsig, askew):
        tipmag = self.tipdistr[np.random.randint(self._len_tipdistr)]

        sorti = np.argsort(self.magdata)
        idxs = sorti[np.in1d(sorti, np.where(self.magdata > tipmag)[0])]
        msk = idxs[:self.nstarsbelow]
        assert len(self.magdata[msk]) == self.nstarsbelow
        assert np.all(self.magdata[msk] > tipmag)

        sig = np.hypot(self.colorunc[msk], colorsig)
        x = (self.colordata[msk]-colorcen)/sig
        lnpnorm = -0.5*(x**2 + np.log(sig))
        lnpskew = np.log1p(erf(askew*x*2**-0.5))
        return lnpnorm + lnpskew, tipmag

