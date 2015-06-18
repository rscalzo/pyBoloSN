#!/usr/bin/env python 

# ============================================================================
# RS 2013/01/24:  Rewriting SN Ia Monte Carlo Markov Chain Reconstruction
# ----------------------------------------------------------------------------
# Now rewriting stuff I wrote in C++ some time ago, in the hope of maybe
# releasing it as a public useful code.
# ============================================================================


# ----------------------------------------------------------------------------
#                         Metric crap-ton of modules
# ----------------------------------------------------------------------------

import sys
import numpy as np
import matplotlib.pyplot as pypl
from BoloMass.Utils import VerboseMsg

from BoloMass.SNPhysics import Density
# from BoloMass.Jeffery99 import Qint
from BoloMass.Jeffery99 import QGP

# ----------------------------------------------------------------------------
#                       Class and function definitions
# ----------------------------------------------------------------------------

def rho_exp(v):
    """Baseline exponential density profile"""
    return np.exp(-v)

def rho_pow7(v):
    """Baseline power-law density profile"""
    return 1.0/(1 + v**7)

def rho_pow3x3(v):
    """Modified power-law density profile tuned to match MPA models"""
    return (1 + v**3)**-3

def run_J99_qint_grid(pklfname, plot=True):
    """Constructs a GP interpolator based on a simple grid of values
    
    Uses the standard Density parametrization (mNilo, mNihi, aNi), i.e.,
    56Ni lives within some range of velocities with some level of mixing.
    """

    density = Density(rho_v=rho_pow3x3)
    # qint = Qint(density)
    qgp = QGP()
    # Sample a grid on the three 56Ni parameters:  mNilo, mNihi, aNi.
    # These will also be the GP hyperparameters.  Then fit the GP.
    # x, y, yerr = qint.qsample(verbose=True)
    mNilo, mNihi, aNi, y, yerr = np.loadtxt(
            "Jeffery99/j99_pow3x3_qint.txt", unpack=True)
    x = np.atleast_2d([mNilo, mNihi, aNi]).T
    qgp.fit(x, y, [0.1, 0.1, 0.1])
    qgp.save(pklfname)
    if not plot:  return

    # Check how we did by plotting the answers for representative values.
    for aNi, c in zip((0.01, 0.07, 0.14, 0.35),
                      ('red', 'orange', 'green', 'blue')):
        for mNihi in np.arange(0.1, 1.01, 0.1):
            # Only print one legend for the lot
            if abs(mNihi - 1.0) < 1e-2:
                label = 'aNi = {0}'.format(aNi)
            else:
                label = '_nolegend_'
            # Pick out x, y values for this slice
            xplot, yplot, yerrplot = [ ], [ ], [ ]
            for xi, yi, yerri in zip(x, y, yerr):
                if abs(xi[2] - aNi) < 1e-3 and abs(xi[1] - mNihi) < 1e-3:
                    xplot.append(xi)
                    yplot.append(yi)
                    yerrplot.append(yerri)
            # Plot integrated points with error bars, plus GP as solid curves
            xplot = np.array(xplot)
            yplot = np.array(yplot)
            yerrplot = np.array(yerrplot)
            pypl.errorbar(xplot[:,0], yplot, yerr=yerrplot,
                          marker='o', ls='None', c=c, label='_nolegend_')
            # Pick some intermediate points so we see how the GP is doing
            xplot = np.atleast_2d([[mNilo, mNihi, aNi]
                                  for mNilo in np.arange(-0.5, mNihi, 0.01)])
            yplot = qgp.eval(xplot)
            pypl.plot(xplot[:,0], yplot, c=c, ls='-', label=label)
    # Now plot an answer not strictly on the grid
    aNi, mNilo, c = 0.2, 0.0, 'violet'
    for mNihi in np.arange(0.1, 1.01, 0.1):
        if abs(mNihi - 1.0) < 1e-2:
            label = 'aNi = {0}'.format(aNi)
        else:
            label = '_nolegend_'
        xplot = np.array([(mNilo, mNihi, aNi)
                         for mNilo in np.arange(0.0, mNihi, 0.1)])
        pypl.plot(xplot[:,0], qgp.eval(xplot), c=c, ls='-', label=label)
    # legend and show
    pypl.xlabel("mnilo")
    pypl.ylabel("Q")
    pypl.legend()
    pypl.show()


if __name__ == "__main__":
    run_J99_qint_grid(sys.argv[1])
