#!/usr/bin/env python 

"""
RS 2013/07/09:  Light Curve Models from Jeffery 1999
"""

# ----------------------------------------------------------------------------
#                                Dependencies
# ----------------------------------------------------------------------------

import numpy as np
import cPickle as pickle
from os.path import dirname
from pygsl.monte import vegas, gsl_monte_function
from pygsl.rng import mt19937_1999
from ..Utils import VerboseMsg

# ----------------------------------------------------------------------------
#                              Package globals
# ----------------------------------------------------------------------------

# VEGAS Monte Carlo integrator.  In the bad old days this had to be a package
# global so that the old SNDensity class was picklable and could be sampled
# by emcee.  But we've decoupled all that now so we don't care so much!
vegas = vegas(3)
# Random number generator (same deal)
rng = mt19937_1999()

# ----------------------------------------------------------------------------
#                       Class and function definitions
# ----------------------------------------------------------------------------

class Qint(object):
    """Monte Carlo integrator to calculate q"""

    def __init__(self, density):
        self.density = density

    def _qint_numerator(self, args, pars):
        """Evaluates the numerator of the q-integral from Jeffery 1999"""
        self._ncalls += 1
        z, zs, mu = args
        # Evaluate density at point z
        rho_z = self.density.rho_bar(z)
        # Evaluate density at point z' = z + zs (vector sum)
        zp = np.sqrt(z**2 + zs**2 + 2*z*zs*mu)
        rho_zp = self.density.rho_bar(zp)
        # Evaluate 56Ni mass fraction at point z
        XNi_z = self.density.XNi_z(z)
        # print "z, zp, rho_z, rho_zp, XNi_z =", z, zp, rho_z, rho_zp, XNi_z
        value = 0.5 * z**2 * XNi_z * rho_z * rho_zp
        # print "value ({0}, {1}, {2}) = {3}".format(z, zs, mu, value)
        return value

    def qint(self):
        """Evaluates the q integral via Monte Carlo"""
        self._ncalls = 0
        # Integration variables are:  z, zs, mu
        bounds_lo = [  0,  0, -1]
        bounds_hi = [ 20, 20, +1]
        qfunc = gsl_monte_function(self._qint_numerator, None, 3)
        qnum = vegas.integrate(qfunc, bounds_lo, bounds_hi, 20000, rng)
        qdenom = np.float64(self.density.fNi)
        return qnum/qdenom

    def qsample(self, plist=None, verbose=True):
        """Evaluates q while changing the 56Ni parameters

        Changes the 56Ni distribution in a density profile without changing
        the profile itself (assumed dimensionless, M and Ekin factored out).
        plist is a list of parameter tuples to be passed to the SNDensity
        instance's set_XNi_pars() method; in the baseline class variant these
        are mNi_lo, mNi_hi, and aNi, corresponding to the inner and outer limits
        of the 56Ni distribution and the mixing scale (Lagrangian coordinates).
        """
        vmsg = VerboseMsg(prefix="Qint.qsample", verbose=verbose)
        vmsg("starting")
        # Some identifying information
        vmsg("Running for basic exponential density profile")
        vmsg("Columns:  aNi mNi_lo mNi_hi q q_err")
        # Okay!  Here we go.
        if plist is None:
            plist = self.density.XNi_pars_sample()
        xfit, yfit, yfiterr = plist, [ ], [ ]
        for p in plist:
            # Set up the 56Ni distribution and calculate q
            self.density.set_XNi_pars(*p)
            myq, myqerr = self.qint()
            if verbose:
                print "{0} {1:.5f} {2:.5f}".format(p, myq, myqerr)
            yfit.append(myq)
            yfiterr.append(myqerr)
        return xfit, yfit, yfiterr
