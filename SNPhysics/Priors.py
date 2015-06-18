#!/usr/bin/env python 

"""
RS 2013/07/09:  Common SN Physics Models and/or Priors

This package contains several useful functions from single papers which are
relevant to SN modeling while not necessarily themselves providing entire
self-consistent models of SN explosions.
"""

# ----------------------------------------------------------------------------
#                               Dependencies
# ----------------------------------------------------------------------------

import numpy as np
from ..Utils.Exceptions import BadInputError

# ----------------------------------------------------------------------------
#                           Function definitions
# ----------------------------------------------------------------------------

def yl05_bind(MWD, lpc):
    """Binding energies from Yoon & Langer 2005, equation (32).

    The central density is really a function of the angular momentum of a
    differentially rotating white dwarf.  Should re-read YL05 to make sure
    I understand what's really going on here.  Input parameters:
        MWD:  mass of the white dwarf (solar masses)
        lpc:  log10 of central density
    Output binding energy in erg.
    """

    # If central density is zero or less, we can't take the log
    if lpc <= 0:
        raise BadInputError("log central density should be > 0")
    # Mass of non-rotating white dwarf with same central density
    MNR = 1.436*(1 - np.exp(-0.01316*np.exp(2.706*np.log(lpc)) + 0.2493*lpc))
    # Rotation supports *more* massive white dwarfs, so MWD > MNR always.
    # If MWD < MNR, tell MH_proposal what MNR is so we can get back in bounds.
    if MWD < MNR:
        raise BadInputError("MWD = {0:.3f} is less than non-rotating value "
                            "{1:.3f} for lpc = {2:.3f}".format(MWD, MNR, lpc))
    # Otherwise:
    benr = -32.759747 + 6.7179802*lpc - 0.28717609*lpc*lpc
    c5 = -370.73052 + lpc*(132.97204 + lpc*(-16.117031 + lpc*0.66986678))
    return 1e+50*(benr + c5*np.exp(1.03*np.log(MWD - MNR)))

def mi09_Enuc(MFe, MNi, MSi, MCO):
    """Energy released via nuclear burning from Maeda & Iwamoto 2009
    
    Input a composition, with masses of elements in solar masses:
        MFe:  mass of stable Fe-peak elements like Cr, Ti, Fe, Co, Ni
        MNi:  mass of 56Ni
        MSi:  mass of intermediate-mass elements like Si, S, Mg, Ca
        MCO:  mass of unburned C + O
    Output energy released via nuclear burning in erg.
    """
    return 1e+51*(1.74*MFe + 1.56*MNi + 1.24*MSi)

def krueger12_eta(lpc):
    """Ratio of 56Ni to total iron-peak yield from Krueger+ 2012

    Fitting formula for K12 central density results.  Based on looking at
    iron-peak yields from Khokhlov, Mueller & Hoflich 1993, I assign a
    flat prior eta = 0.9 below a central density of 1e+9 g/cm^3.
    Could probably do better by incorporating other results e.g. from the
    MPA group (Seitenzahl+ 2013).  Input lpc (log10 of central density),
    output eta = MNi/(MNi + MFe).
    """
    pc9 = 1e-9 * 10**lpc
    return min(0.95, 0.95 - 0.05*pc9), max(0.025, 0.03*pc9)
