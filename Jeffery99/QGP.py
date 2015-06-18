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
from ..Utils import VerboseMsg

# stupid kludge b/c I don't know how else to override system packages when
# they keep clobbering my $PYTHONPATH
# import sys
# sys.path = ['/home/rscalzo/local/lib/python2.7/site-packages'] + sys.path
import sklearn
print "Using sklearn version", sklearn.__version__, "from", sklearn.__file__

from BoloMass.SNPhysics import Density

# ----------------------------------------------------------------------------
#                              Package globals
# ----------------------------------------------------------------------------

# Compton scattering opacity for 56Co gamma rays in the optically thin limit
kgamma = 0.025 # cm^2/g

# Pickle file for standard package Gaussian Process Q-factor interpolator
_datadir = dirname(__file__)
# stdQgp_exp_pklfn = _datadir + "/j99_exp_qgp.pkl"
# stdQgp_pow3x3_pklfn = _datadir + "/j99_pow3x3_qgp.pkl"
stdQgp_exp_pklfn = _datadir + "/j99_exp_qgp_mlo.pkl"
stdQgp_pow3x3_pklfn = _datadir + "/j99_pow3x3_qgp_mlo.pkl"

# ----------------------------------------------------------------------------
#                       Class and function definitions
# ----------------------------------------------------------------------------

def t0(q, M, Ekin):
    """Time in days until a mean optical depth of 1 to Compton scattering
   
    Dependencies now encoded in a function so we can abstract them away when
    running MCMC.  It's cleaner this way.
        q:  Jeffery q factor
        M:  mass in grams
        Ekin:  kinetic energy in ergs
    """
    my_v_sc, my_rho_sc = Density.v_sc(M, Ekin), Density.rho_sc(M, Ekin)
    try:
        return np.sqrt(my_v_sc * my_rho_sc * kgamma * q) / 86400.0
    except Exception as e:
        print "FAIL with inputs:  ", q, M, Ekin
        raise e


class QGP(object):
    """Gaussian process interpolator for Q values for a given model"""

    def __init__(self, pklfname=None):
        if pklfname is None:
            self.gp = None
        else:
            with open(pklfname) as pklfile:
                self.gp = pickle.load(pklfile)

    def save(self, pklfname):
        """Saves GP to a pickle file"""
        with open(pklfname, 'w') as pklfile:
            pickle.dump(self.gp, pklfile)

    def fit(self, x, y, ll0, llL=None, llU=None):
        """Fits a GP to existing tuples of parameters"""
        assert(len(x) == len(y))
        x = np.atleast_2d(x)
        y = np.array(y)
        ll0 = np.array(ll0)
        # The inputs ought to be known to precision about 1e-3 or better
        nugget = 1.0e-3
        # Convenience setup for diagonal squared exponential
        theta0 = 0.5/ll0**2
        thetaL, thetaU = 0.1*theta0, 10*theta0
        self.gp = sklearn.gaussian_process.GaussianProcess(
                theta0=theta0, thetaL=thetaL, thetaU=thetaU,
                storage_mode='light', nugget=nugget,
                corr="squared_exponential", regr='quadratic', verbose=True)
        self.gp.fit(x, y)

    def eval(self, x):
        """Evaluates the GP using tuples of the same form as the fit"""
        # We don't care what the errors are, they should be sub-dominant
        if self.gp is None:
            raise Exception("EPIC FAIL:  you need to set up the GP first!")
        else:
            return self.gp.predict(np.atleast_2d(x))


# Package global Gaussian Process Q-factor interpolators
stdQgp_exp = QGP(stdQgp_exp_pklfn)
stdQgp_pow3x3 = QGP(stdQgp_pow3x3_pklfn)
