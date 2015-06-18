#!/usr/bin/env python 

"""
RS 2013/08/23:  Wrapper class for marginalizing over reddening
"""

# ----------------------------------------------------------------------------
#                                Dependencies
# ----------------------------------------------------------------------------

import numpy as np
from ..Utils import BadInputError
from .SNMCMC import SNMCMC

# ----------------------------------------------------------------------------
#                       Class and function definitions
# ----------------------------------------------------------------------------

def SNMCMC_EBMV(baseMCMC):
    """Class template for marginalizing over reddening
    
    This function builds a new SNMCMC subclass, based on any pre-existing
    subclass, which treats host galaxy reddening by averaging over reddened
    light curves (essentially the way it's done in Scalzo+ 2012).
    """

    if not issubclass(baseMCMC, SNMCMC):
        raise BadInputError("{0} is not a subclass of SNMCMC".format
                            (baseMCMC.__name__))

    class MyClass(baseMCMC):

        _features = dict(baseMCMC._features)
        _features.update({
           # name,     def,  blo,  hlo,  hhi,  bhi,  res,  fmt, label
           'ebmv':   [ 0.05,  0.0,  0.0,  0.4,  0.4, 0.005, "{0:8.3f}",
                      "Host galaxy E(B-V) (mag)"],
        })
        _mcmcpars = list(baseMCMC._mcmcpars) + ['ebmv']
        _confints = list(baseMCMC._confints) + ['ebmv']
        _init_kwdef = dict(baseMCMC._init_kwdef)
        _init_kwdef.update({ 'ebmv':   0.00, 'ebmv_err': 0.01 })

        def __init__(self, t, multi_L, multi_dL, multi_ebmv, **init_kwargs):
            """Initialize with a series of light curves at different reddenings

            Below, n is the number of light curve points and M is the number of
            discrete values of E(B-V)_host over which we are sampling, assuming
            we've corrected for Milky Way dust already using something like the
            SFD Galactic dust maps.
               t:  shape = (n, ) time since bolometric maximum light, in days
               L:  shape = (M, n) bolometric luminosity in erg/s
               dL:  shape = (M, n) bolometric luminosity 1-sigma error in erg/s
               ebmv:  shape = (M, ) discrete reddening values
            """
            # Unpack kwargs
            kw = dict(self._init_kwdef)
            kw.update(init_kwargs)
            # First set up the internal data members.
            self.t = np.atleast_1d(t)
            self._multi_L = np.atleast_2d(multi_L)
            self._multi_dL = np.atleast_2d(multi_dL)
            self._multi_ebmv = np.atleast_1d(multi_ebmv)
            self.ebmv_Pmu, self.ebmv_Psig = kw['ebmv'], kw['ebmv_err']
            ebmv_av = 0.50*(multi_ebmv[-1] + multi_ebmv[0])
            ebmv_ss = 0.05*(multi_ebmv[-1] - multi_ebmv[0])
            self._features['ebmv'][0:6] = [
                self.ebmv_Pmu, multi_ebmv[0], multi_ebmv[0],
                multi_ebmv[-1], multi_ebmv[-1], ebmv_ss]
            # Find the light curve with reddening closest to the prior.
            i = abs(self._multi_ebmv - kw['ebmv']).argmin()
            # Initialize the base class.  The indices of what to fit and what
            # not to fit currently depend only on time, which works in the same
            # way as the baseline class; we will have to get fancier if we want
            # to incorporate anything like outlier rejection.
            baseMCMC.__init__(self, t, multi_L[i], multi_dL[i], **kw)

        def logl(self, pars, blob=None):
            """Log likelihood for MCMC including reddening
            
            This is a wrapper for the original baseMCMC.lnprob() call.
            It fills the internal data members self.Lfit and self.dLfit with
            linearly interpolated values typical of this reddening value.
            """

            # Unpack parameters from vector
            ebmv = pars[-1]

            # The sampler bounds checker should make sure E(B-V) is in range,
            # but if it isn't, cope with it now.
            x, _x, nx = ebmv, self._multi_ebmv, len(self._multi_ebmv)
            if not _x[0] <= x < _x[-1]:
                raise BadInputError("reddening E(B-V) = {0:.3f} out of bounds "
                                    "(range = [{1:.3f}, {2:.3f}])".format
                                    (x, _x[0], _x[-1]))
            if not len(_x) == len(self._multi_L) == len(self._multi_dL):
                raise BadInputError("{0} reddening values provided "
                                    "for {1} corresponding light curves "
                                    "and {2} sets of error bars".format(len(_x),
                                    len(self._multi_L), len(self._multi_dL)))
            # Linearly interpolate between the light curves.  Assume
            # self._multi_ebmv is sorted in order of increasing reddening.
            # Consider neighboring values of reddening to be 100% correlated.
            i0 = max([i for i in range(nx) if _x[i] <= ebmv])
            fi = (x - _x[i0]) / (_x[i0+1] - _x[i0])
            _y = (1-fi) * self._multi_L[i0] + fi * self._multi_L[i0+1]
            _yerr = (1-fi) * self._multi_dL[i0] + fi * self._multi_dL[i0+1]
            idx, nidx = self.ifit, self.infit
            self.Lfit, self.Lnfit = _y[idx], _y[nidx]
            self.dLfit, self.dLnfit = _yerr[idx], _yerr[nidx]

            # Return the likelihood and blob at this E(B-V)
            return baseMCMC.logl(self, pars[:-1])

        def logp(self, pars, blob=None):
            """Log prior for MCMC including reddening"""
            chpri = ((pars[-1] - self.ebmv_Pmu)/self.ebmv_Psig)**2
            return baseMCMC.logp(self, pars[:-1]) - 0.5*chpri

        def fillblob(self, pars):
            """Fills a blob with everything but likelihood and prior"""
            blob = baseMCMC.fillblob(self, pars[:-1])
            blob.update({ 'ebmv': pars[-1] })
            return blob

    return MyClass
