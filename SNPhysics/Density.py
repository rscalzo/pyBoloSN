#!/usr/bin/env python 

"""
RS 2013/07/09:  Supernova Ejecta Density Representation

This package contains a wrapper class for representations of a dimensionless
density profile of a type I SN, powered by the radioactive decay of 56Ni.
It also contains a few standard functions for introducing the units back
into those profiles in ways which preserve invariants, such as the mass and
kinetic energy of the ejecta.
"""

# ----------------------------------------------------------------------------
#                         Metric crap-ton of modules
# ----------------------------------------------------------------------------

import numpy as np
from scipy.integrate import quad
from scipy.special import erfc
from ..Utils.VerboseMsg import VerboseMsg

# ----------------------------------------------------------------------------
#                       Class and function definitions
# ----------------------------------------------------------------------------

def _part(x, x0, sig):
    """Partition of unity, 1 at low values and 0 at high ones"""
    return 0.5*erfc((x-x0)/(sig*np.sqrt(2.0)))


class Density(object):
    """Semi-analytic representation of ejecta density."""

    @staticmethod
    def v_sc(M, Ekin):
        """Consistent definition of scaling velocity"""
        return np.sqrt(2*Ekin/M)

    @staticmethod
    def rho_sc(M, Ekin):
        """Consistent definition of scaling density"""
        return M/(4*np.pi * Density.v_sc(M, Ekin)**3)

    def __init__(self, rho_v=(lambda v: np.exp(-v)), verbose=True):
        """Initialize parametrization of density function

        rho_v:  user-defined function of velocity v, returning density.
            (Recommended that this accept numpy.array inputs.)
        verbose:  print status messages for debugging?  (bool)
        """
        # Density normalization factors (we'll solve for these in a bit)
        self._v0, self._rho0, self._XNi0= 1, 1, 1
        vmsg = VerboseMsg(prefix="Density.__init__",
                          verbose=verbose, flush=False)
        # First find the actual mass and kinetic energy of the model by
        # integrating the density.
        self.rho = rho_v
        M = 4*np.pi * self.Mrho(2, np.inf)
        Ekin = 2*np.pi * self.Mrho(4, np.inf)
        vmsg("original:  M, Ekin =", M, Ekin)
        # Set the normalization factors so that rho_bar is dimensionless,
        # with M = Ekin = 1 in the new units.
        self._v0 = Density.v_sc(M, Ekin)
        self._rho0 = Density.rho_sc(M, Ekin)
        vmsg("self._v0, self._rho0 =", self._v0, self._rho0)
        M_bar = self.Mrho(2, np.inf)
        Ekin_bar = 0.5*self.Mrho(4, np.inf)
        vmsg("in new units:  M_bar, Ekin_bar =", M_bar, Ekin_bar)
        # 56Ni distribution; set XNi0 so it integrates to the right fNi.
        self.set_XNi_pars(0.1, 0.9, 0.1, 0.4)
        vmsg("self._XNi0 =", self._XNi0)
        vmsg("self.fNi =", self.Mrho(2, np.inf, w=self.XNi_z))

    def set_XNi_pars(self, mNi_lo, mNi_hi, aNi, fNi=None):
        """Set parameters regarding the distribution of 56Ni in the ejecta"""
        self._XNi0 = 1
        self.mNi_lo, self.mNi_hi, self.aNi = mNi_lo, mNi_hi, aNi
        MXNi = self.Mrho(2, np.inf, w=self.XNi_z)
        if fNi:
            if fNi < MXNi:
                self.fNi, self._XNi = fNi, fNi/MXNi
            else:
                print "set_XNi_pars:  WARNING, result is invalid!"
                self._XNi0 = 1
        else:
            self.fNi, self._XNi0 = MXNi, 1

    def rho(self, v):
        """Default density profile in g/(cm/s)^3 ("physical units")"""
        return np.exp(-v)

    def rho_bar(self, z):
        """Dimensionless density profile where z = v/v_sc

        NB!!!:  Once we scale rho into non-dimensional units, we're done.
        Everything else can be done in z units where z = v/vKE.
        For hydrodynamic models, this scaling probably has to be done before
        the density profile is input, since leaving them in physical values
        (which are large in cgs units) may cause numerical problems computing
        some of the other quantities later on.
        """
        return self.rho(z*self._v0) / self._rho0

    def Mrho(self, n, z, w=(lambda z: 1)):
        """Return an integrated moment of the density"""
        a, z = [ ], np.array(z)
        ig = lambda zp: zp**n * w(zp) * self.rho_bar(zp)
        for zi in z.ravel():  a.append(quad(ig, 0, zi)[0])
        return np.reshape(a, z.shape)

    def Mrhoc(self, n, z, w=(lambda z: 1)):
        """Return an integrated moment of the density"""
        a, z = [ ], np.array(z)
        ig = lambda zp: zp**n * w(zp) * self.rho_bar(zp)
        for zi in z.ravel():  a.append(quad(ig, zi, np.inf)[0])
        return np.reshape(a, z.shape)

    def XNi(self, m):
        """56Ni fraction in ejecta as a function of enclosed mass fraction"""
        m = np.array(m)
        return self._XNi0 *((1.0 - _part(m, self.mNi_lo, self.aNi))
                                 * _part(m, self.mNi_hi, self.aNi))

    def XNi_z(self, z):
        """56Ni fraction in ejecta as a function of dimensionless velocity"""
        mz = self.Mrho(2, z)
        return self.XNi(mz)

    def XNi_pars_sample(self):
        """Sample a parameter grid representing different 56Ni distributions
        
        This generates parameters to set_XNi_pars() which can then be used
        to calculate q within the context of a driver script.  We can also,
        of course, use MCMC or similar to sample parameters if given bounds
        or a prior on different density parametrizations.  This version just
        uses a simple grid.  Returns a np-ified version suitable for use when
        fitting a Gaussian process.
        """

        XNiP = [ ]
        for aNi in (0.01, 0.07, 0.14, 0.20, 0.35):
            for mNi_lo in np.arange(0, 0.91, 0.1):
                for mNi_hi in np.arange(mNi_lo + 0.1, 1.01, 0.1):
                    XNiP.append([mNi_lo, mNi_hi, aNi])
        return np.array(XNiP)
