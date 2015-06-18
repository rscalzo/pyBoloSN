#!/usr/bin/env python 

"""
RS 2014/02/03:  Wrapper class for marginalizing over E(B-V) and R_V
"""

# ----------------------------------------------------------------------------
#                                Dependencies
# ----------------------------------------------------------------------------

import numpy as np
from scipy import linalg
from ..Utils import BadInputError
from .SNMCMC import SNMCMC

# ----------------------------------------------------------------------------
#                       Class and function definitions
# ----------------------------------------------------------------------------

def SNMCMC_EBMVwRV(baseMCMC):
    """Class template for marginalizing over reddening and extinction law
    
    This function builds a new SNMCMC subclass, based on any pre-existing
    subclass, which treats host galaxy reddening by averaging over reddened
    light curves (essentially the way it's done in Scalzo+ 2012).
    The update from SNMCMC_EBMV is that now we also marginalize over the
    reddening law R_V, by bilinear interpolation.
    """

    if not issubclass(baseMCMC, SNMCMC):
        raise BadInputError("{0} is not a subclass of SNMCMC".format
                            (baseMCMC.__name__))

    class MyClass(baseMCMC):

        _features = dict(baseMCMC._features)
        _features.update({
           # name,     def,  blo,  hlo,  hhi,  bhi,  res,  fmt, label
           'ebmv':   [ 0.05,  0.0,  0.0,  0.5,  0.4, 0.005, "{0:8.3f}",
                      "Host galaxy E(B-V) (mag)"],
           'Rv':     [  3.1,  0.0,  0.0,  9.5,  9.5,   0.5, "{0:8.3f}",
                      "Host galaxy R_V (mag/mag)"],
        })
        _mcmcpars = list(baseMCMC._mcmcpars) + ['ebmv', 'Rv']
        _confints = list(baseMCMC._confints) + ['ebmv', 'Rv']
        _init_kwdef = dict(baseMCMC._init_kwdef)
        _init_kwdef.update({ 'ebmv': 0.00, 'ebmv_err': 0.01,
                             'Rv':   3.1,  'Rv_err':   0.1,
                             'covEE': 0.0, 'covER': 0.0, 'covRR': 0.0 })

        @staticmethod
        def _ifi(x, _x):
            """Gives fractional indices for linear interpolation"""
            # Bounds check
            if not _x[0] <= x < _x[-1]:
                raise IndexError("x = {1:.3f} out of bounds "
                                 "(range = [{2:.3f}, {3:.3f}])".format
                                 (x, _x[0], _x[-1]))
            i0 = max([i for i in range(len(_x)) if _x[i] <= x])
            fi = (x - _x[i0]) / (_x[i0+1] - _x[i0])
            return i0, fi

        def _idx(self, i_ebmv, i_Rv):
            """Gives indices in internal representation for given E(B-V), R_V

            Because we've got two independent variables here (E(B-V) and R_V),
            the cleanest way to go is to tabulate only on regular grids for
            both parameters and to specify an indexing scheme from the outset.
            """
            return i_ebmv + i_Rv*len(self._multi_ebmv)

        def __init__(self, t, multi_L, multi_dL,
                     multi_ebmv, multi_Rv, **init_kwargs):
            """Initialize with a series of light curves at different reddenings

            Below, n is the number of light curve points and M is the number of
            discrete values of E(B-V)_host over which we are sampling, assuming
            we've corrected for Milky Way dust already using something like the
            SFD Galactic dust maps.
               t:  shape = (n, ) time since bolometric maximum light, in days
               L:  shape = (M*N, n) bolometric luminosity in erg/s
               dL:  shape = (M*N, n) bolometric luminosity 1-sigma error in erg/s
               ebmv:  shape = (M, ) discrete reddening values
               Rv:    shape = (N, ) discrete extinction law slope values
            """
            # Unpack kwargs
            kw = dict(self._init_kwdef)
            kw.update(init_kwargs)
            # First some sanity checks.
            self.t = np.atleast_1d(t)
            self._multi_L = np.atleast_2d(multi_L)
            self._multi_dL = np.atleast_2d(multi_dL)
            self._multi_ebmv = np.atleast_1d(multi_ebmv)
            self._multi_Rv = np.atleast_1d(multi_Rv)
            if self._multi_L.shape != (len(multi_ebmv)*len(multi_Rv), len(t)):
                raise BadInputError(
                    "multi_L.shape = {0} != ({1}*{2}, {3}) "
                    "as (ebmv, Rv) indexing scheme requires".format
                    (self._multi_L.shape, len(multi_ebmv), len(multi_Rv), len(t)))
            if self._multi_dL.shape != self._multi_L.shape:
                raise BadInputError(
                    "multi_dL.shape = {0} != {1} = multi_L.shape".format
                    (multi_L.shape, multi_dL.shape))
            # Now set up the rest of the internal data members.
            self.ebmv_Pmu, self.ebmv_Psig = kw['ebmv'], kw['ebmv_err']
            self.Rv_Pmu,   self.Rv_Psig =   kw['Rv'],   kw['Rv_err']
            self.covEE = kw['covEE']
            self.covER = kw['covER']
            self.covRR = kw['covRR']
            self.cov_ebmv_Rv_inv = np.matrix(linalg.inv(
                [[kw['covEE'], kw['covER']], [kw['covER'], kw['covRR']]]))
            self.p0_ebmv_Rv = np.atleast_2d([kw['ebmv'], kw['Rv']])
            ebmv_av = 0.50*(multi_ebmv[-1] + multi_ebmv[0])
            ebmv_ss = 0.05*(multi_ebmv[-1] - multi_ebmv[0])
            Rv_av = 0.50*(multi_Rv[-1] + multi_Rv[0])
            Rv_ss = 0.05*(multi_Rv[-1] - multi_Rv[0])
            self._features['ebmv'][0:6] = [
                self.ebmv_Pmu, multi_ebmv[0], multi_ebmv[0],
                multi_ebmv[-1], multi_ebmv[-1], ebmv_ss]
            self._features['Rv'][0:6] = [
                self.Rv_Pmu, multi_Rv[0], multi_Rv[0],
                multi_Rv[-1], multi_Rv[-1], Rv_ss]
            # Make sure critical data members have been set properly
            print "SNMCMC_EBMVwRV:  Internal parameter check..."
            for attr in ['ebmv_Pmu', 'ebmv_Psig', 'Rv_Pmu', 'Rv_Psig',
                         'covEE', 'covER', 'covRR',
                         'p0_ebmv_Rv', 'cov_ebmv_Rv_inv']:
                print ">> {0:<12} {1:>20}".format(attr, getattr(self, attr))
            # Find the light curve with reddening closest to the prior.
            i_ebmv, fi_ebmv = self._ifi(kw['ebmv'], self._multi_ebmv)
            i_Rv,   fi_Rv   = self._ifi(kw['Rv'],   self._multi_Rv)
            i = self._idx(i_ebmv, i_Rv)
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
            ebmv, Rv = pars[-2:]

            # Linearly interpolate between the light curves.
            # Consider neighboring values of reddening to be 100% correlated.
            i1, fi1 = self._ifi(ebmv, self._multi_ebmv)
            i2, fi2 = self._ifi(Rv,   self._multi_Rv)
            idx = self._idx
            _y = ((1-fi1)*(1-fi2) * self._multi_L[idx(i1,   i2  )] +
                     fi1 *(1-fi2) * self._multi_L[idx(i1+1, i2  )] +
                  (1-fi1)*   fi2  * self._multi_L[idx(i1,   i2+1)] +
                     fi1 *   fi2  * self._multi_L[idx(i1+1, i2+1)])
            _yerr = ((1-fi1)*(1-fi2) * self._multi_dL[idx(i1,   i2  )] +
                        fi1 *(1-fi2) * self._multi_dL[idx(i1+1, i2  )] +
                     (1-fi1)*   fi2  * self._multi_dL[idx(i1,   i2+1)] +
                        fi1 *   fi2  * self._multi_dL[idx(i1+1, i2+1)])
            idx, nidx = self.ifit, self.infit
            self.Lfit, self.Lnfit = _y[idx], _y[nidx]
            self.dLfit, self.dLnfit = _yerr[idx], _yerr[nidx]

            # Return the likelihood and blob at this E(B-V)
            return baseMCMC.logl(self, pars[:-2])

        def logp(self, pars, blob=None):
            """Log prior for MCMC including reddening"""
            Cinv = self.cov_ebmv_Rv_inv
            dp = np.atleast_2d(pars[-2:]) - self.p0_ebmv_Rv
            chpri = np.float(dp * Cinv * dp.T)
            return baseMCMC.logp(self, pars[:-2]) - 0.5*chpri

        def fillblob(self, pars):
            """Fills a blob with everything but likelihood and prior"""
            blob = baseMCMC.fillblob(self, pars[:-2])
            blob.update({ 'ebmv': pars[-2], 'Rv': pars[-1] })
            return blob

    return MyClass
