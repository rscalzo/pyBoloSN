#!/usr/bin/env python 

"""
RS 2013/07/09:  Light Curve Models from Arnett 1980, 1982
"""

# ----------------------------------------------------------------------------
#                               Dependencies
# ----------------------------------------------------------------------------

import sys
import numpy as np
import cPickle as pickle
from os.path import dirname
from scipy.integrate import quad
from sklearn import gaussian_process

# ----------------------------------------------------------------------------
#                             Package globals
# ----------------------------------------------------------------------------

# e-folding decay times of 56Ni and 56Co, in days
tNi, tCo = 8.8, 111.3
# decay constants of 56Ni and 56Co in days^-1
lNi, lCo = 1.0/tNi, 1.0/tCo
# decay energies in MeV
QNiG, QCoG, QCoE = 1.75, 3.61, 0.12

# Working out the normalization constant for epsilon:
# We have energy released in MeV day^-1 (atom of 56Ni)^-1, where we really
# want it expressed in erg s^-1 (solar masses of 56Ni)^-1.  So:
# 2e+33 kg/(56 AMU * 1.67e-27 kg) = 2.14e+55 Ni atoms / Msol
# 1 MeV/day = 1.85e-11 erg/s
# thus:  1.85e-11 (erg/s)/(MeV/day) * 2.14e+55 atoms/Msol
epsilon_norm = 3.96e+44

# Pickle file for standard package Gaussian Process light curve interpolator
_datadir = dirname(__file__)

# ----------------------------------------------------------------------------
#                           Function definitions
# ----------------------------------------------------------------------------

def epsilon(t, tg):
    """Energy *deposited* via radioactive decay from Nadyozhin 1994

    Calculates luminosity of radioactive decay in erg s^-1 (MNi/Msol)^-1,
    as a function of time t in days since explosion.  In other words,
    multiply this by the nickel mass in solar masses to get the luminosity.
        t:  time since explosion in days
        tg:  fiducial time till optical depth = 1 to 56Co gamma rays,
             in days after explosion (Jeffery 1999 calls this t_0)
    """

    # cast time as numpy array
    t = np.atleast_1d(t)

    # Calculate optical depth to 56Co gamma rays.  We assume that positrons
    # remain fully trapped during the phases of interest, but to be rigorous
    # we should probably include a positron optical depth as well.
    tau = (tg/t)**2
    # then input to the parametric form from Stritzinger & Leibundgut 2006
    return (lNi*np.exp(-lNi*t)*QNiG
            + (lCo*(lNi/(lNi-lCo))*(np.exp(-lCo*t) - np.exp(-lNi*t))
            * (QCoE + QCoG*(1-np.exp(-tau))))) * epsilon_norm

def Lambda(t, y):
    """Original Arnett 1982 dimensionless bolometric light curve expression
    
    Calculates the bolometric light curve due to radioactive decay of 56Ni,
    assuming no other energy input.  
        t:  time since explosion in days
        y:  Arnett 1982 light curve width parameter (typical 0.7 < y < 1.4)
    Returns the dimensionless light curve shape function.
    """
    tm = 2*tNi*y
    a, x = [ ], np.atleast_1d(t/tm)
    ig = lambda z: 2*z * np.exp(-2*z*y + z**2)
    for xi in x.ravel():  a.append(np.exp(-xi**2) * quad(ig, 0, xi)[0])
    return np.array(a)

def A82LC_Co(t, y, tg):
    """Modified Arnett law adding Co decay
    
    This version is generalized to include a source term with 56Co.
    Done with reference to Dado & Dar's shameless rip-off, wherein
    t_r (D&D's LC width) = 0.707 * tau_m (A82's LC width).  We work with a
    time axis with units of days rather than dimensionless time, to keep from
    confusing the Gaussian processes (and the user!).
        t:  time since explosion in days
        y:  Arnett 1982 light curve width parameter (typical 0.7 < y < 1.4)
        tg:  fiducial time till optical depth = 1 to 56Co gamma rays,
             in days after explosion (Jeffery 1999 calls this t_0)
    Returns light curve normalized to 1.0 Msol of 56Ni.
    """
    tm = 2*tNi*y                        # Arnett 1982 calls this tau_m
    a, x = [ ], np.atleast_1d(t/tm)
    ig = lambda xp:  np.exp(xp**2) * 2*xp * epsilon(tm*xp, tg)
    for xi in x.ravel():  a.append(np.exp(-xi**2) * quad(ig, 0, xi)[0])
    return np.array(a)

def A82LC_CoR0(t, y, w, tg):
    """Modified Arnett law adding Co decay and finite-size effects

    This version is generalized to include a source term with 56Co as well
    as effects of a non-zero initial size.  It therefore includes P*dV effects
    from energy advected as the supernova shock breaks out through the outer
    layers of the star.  Applicable for SNe Ib/c, or for double-degenerate
    "cold merger" SNe Ia with a substantial C+O envelope.
        t:  time since explosion in days
        y:  Arnett 1982 light curve width parameter (typical 0.7 < y < 1.4)
        w:  Arnett 1982 finite size effect parameter (expect w < 0.2)
        tg:  fiducial time till optical depth = 1 to 56Co gamma rays,
             in days after explosion (Jeffery 1999 calls this t_0)
    Includes *only* the trapped radioactive decay luminosity per unit 56Ni.
    Initial thermal energy from the explosion is done below.
    """
    tm = 2*tNi*y                        # Arnett 1982 calls this tau_m
    a, x = [ ], np.atleast_1d(t/tm)
    # Below:  u = w*x + x**2, du = (w + 2*x)*dx
    ig = lambda xp:  np.exp((w+xp)*xp) * (w+2*xp) * epsilon(tm*xp, tg)
    for xi in x.ravel():
        a.append(np.exp(-(w+xi)*xi) * quad(ig, 0, xi)[0])
    return np.array(a)

def A82LC_EthR0(t, y, w):
    """Diffusion of initial thermal shock energy through envelope

    Since this piece can be calculuated completely analytically, it makes
    little sense to bind it up with the light curve calculation which needs
    at least some quadrature.  Thus this can be done on the fly and doesn't
    need to be represented by a Gaussian process.
        t:  time since explosion in days
        y:  Arnett 1982 light curve width parameter (typical 0.7 < y < 1.4)
        w:  Arnett 1982 finite size effect parameter (expect w < 0.2)
    Returns luminosity per unit initial thermal energy, that is, multiply
    the below by Eth0 to get the full light curve.
    """
    tm = 2*tNi*y                        # Arnett 1982 calls this tau_m
    td = tm/(w + 1e-10)                 # Arnett 1982 calls this tau_0
    a, x = [ ], np.atleast_1d(t/tm)
    # Below:  u = w*x + x**2
    return np.exp(-(w+x)*x) / (td * 86400.0)

def A82LC_full(t, y, w, tg, MNi, Eth0):
    """Full Arnett 1982 LC directly evaluated, including 56Co and finite R0"""
    return MNi * A82LC_CoR0(t, y, w, tg) + Eth0 * A82LC_EthR0(t, y, w)

def tau_h(R0, vsc):
    """Arnett 1982 expansion timescale, in days
    
        R0:  initial radius in cm
        vsc:  scaling velocity in cm/s
    """
    return (R0/vsc) / 86400.0

def tau_0(R0, kappa, M, beta=13.7):
    """Arnett 1982 diffusion timescale, in days
    
        R0:  initial radius in cm
        kappa:  approximate opacity in cm^2/g
        M:  ejected mass in g
        beta:  dimensionless form factor, roughly 13.7 (Arnett 1980, 1982)
    """
    return (kappa*M/(3e+10*beta*R0)) / 86400.0

def tau_m(vsc, kappa, M, beta=13.7):
    """Arnett 1982 light curve width timescale, in days
    
        vsc:  scaling velocity in cm/s
        kappa:  approximate opacity in cm^2/g
        M:  ejected mass in g
        beta:  dimensionless form factor, roughly 13.7 (Arnett 1980, 1982)
    """
    R0 = 1e+6 # cm; not really important
    return np.sqrt(2 * tau_h(R0, vsc) * tau_0(R0, kappa, M, beta=beta))


# ----------------------------------------------------------------------------
#                             Class definitions
# ----------------------------------------------------------------------------


def A82LC_regr(x):
    """Mean function basis for Gaussian Process regression"""
    x = np.asarray(x, dtype=np.float)
    n_eval = x.shape[0]
    # Extract the relevant degrees of freedom
    AA = np.array
    t, tg = AA([x[:,0]]).T, AA([x[:,3]]).T
    fNi = t**2 * np.exp(-t/(2*tNi))
    fCo = t**2 / (1 + (t/(tg + 1e-3))**4)
    f = np.hstack([fNi, fCo])
    return f


class A82LC_gp(object):
    """Class to encapsulate GP interpolation of light curves"""

    def __init__(self, pklfname):
        try:
            self._load_gp(pklfname)
        except:
            self._setup_gp()
            self._save_gp(pklfname)

    def _setup_gp(self):
        """Set up a Gaussian process interpolator
        
        This sets up a GP regression interpolator to evaluate the radioactive
        part of the finite-size thing.
        """
        # Set up the grid
        t = np.array([0, 1, 2] + range(5, 120, 5) + [118, 119, 120],
                     dtype=np.float)
        y = np.arange(0.5, 1.51, 0.25)
        w = np.arange(0.0, 0.51, 0.25)
        tg = np.arange(15.0, 75.1, 15.0)
        # ok, let's go
        X, L = [ ], [ ]
        for yi in y:
            for tgi in tg:
                for wi in w:
                    print "setup:  ", yi, wi, tgi
                    sys.stdout.flush()
                    Lc = A82LC_CoR0(t, yi, wi, tgi)
                    for ti, Lci in zip(t, Lc):
                        X.append([ti, yi, wi, tgi])
                        L.append(Lci)
        # Okay, we've set up the inputs.  Now make stuff happen.
        print "initial thing set up with", len(L), "points"
        print "fitting GP"
        sys.stdout.flush()
        ll0 = np.array([1.0, 1.0, 1.0, 5.0])
        llL, llU = 0.01*ll0, 100.0*ll0
        thetaL, theta0, thetaU = 0.5/llU**2, 0.5/ll0**2, 0.5/llL**2
        self.gp = gaussian_process.GaussianProcess(
                theta0=theta0, thetaL=thetaL, thetaU=thetaU,
                verbose=True, nugget=1e-10, storage_mode='light',
                corr='squared_exponential', regr=A82LC_regr)
        self.gp.fit(X, L)
        # print "GP fit done, theta =", self.gp.theta_
        sys.stdout.flush()

    def _save_gp(self, pklfname):
        """Saves GP to a pickle file"""
        with open(pklfname, 'w') as pklfile:
            pickle.dump(self.gp, pklfile)

    def _load_gp(self, pklfname):
        """Loads GP from a pickle file"""
        with open(pklfname) as pklfile:
            self.gp = pickle.load(pklfile)

    def __call__(self, t, pars):
        """Evaluates the light curve, given the parameters"""
        # Unpack parameters
        t = np.atleast_1d(t).ravel()
        y, w, tg, MNi, Eth0 = pars
        # Evaluate the radiaoctive part via Gaussian process
        X = np.atleast_2d([(ti, y, w, tg) for ti in t])
        lc_Co = self.gp.predict(X)
        # Evaluate the trapped thermal energy
        lc_Eth = A82LC_EthR0(t, y, w)
        # Return the full light curve
        return MNi * lc_Co + Eth0 * lc_Eth


# Package global standard Gaussian Process light curve interpolator
stdLCgp = A82LC_gp(_datadir + "/a82lc_gp_4d.pkl")
