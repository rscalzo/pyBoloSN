from BoloMass.Arnett82 import tau_0, Lambda, A82LC_full, A82LC_gp
from RetroSpect.Plotting import color_ramp

import sys
import numpy as np
import matplotlib.pyplot as pypl


def test_A82_Lambda():
    """Test plots for Lambda"""
    y = np.arange(0.7, 1.41, 0.1)
    c = color_ramp(len(y))
    for yi, ci in zip(y, c):
        pypl.semilogy(t, Lambda(t, yi), color=ci)
    pypl.show()

def test_A82LC_full_01():
    """Test plots for A82LC_full"""
    y, tg, MNi, Eth0 = 1.0, 40.0, 0.6, 0.0e+51
    R0 = np.array([0.0, 0.1, 0.3, 1.0, 3.0, 10.0]) * 1e+14
    c = color_ramp(len(R0))
    for R0i, ci in zip(R0, c):
        td = tau_0(R0i, 0.1, 2.8e+33)
        L0, w = Eth0/(td * 86400), y*17.6/td
        print "R0, tau0, L0, w =", R0i, td, L0, w
        pypl.subplot(2, 1, 1)
        pypl.plot(t, A82LC_full(t, y, w, tg, MNi, Eth0), color=ci)
        pypl.subplot(2, 1, 2)
        pypl.semilogy(t, A82LC_full(t, y, w, tg, MNi, Eth0), color=ci)
    pypl.show()

def test_A82LC_full_02():
    """More test plots for A82LC_full"""
    y, tg, MNi, R0 = 1.0, 40.0, 0.6, 1e+13
    Eth0 = np.arange(0.0, 0.51, 0.1) * 1e+51
    c = color_ramp(len(Eth0))
    for Ethi, ci in zip(Eth0, c):
        td = tau_0(R0, 0.1, 2.8e+33)
        L0, w = Ethi/(td * 86400), y*17.6/td
        print "R0, tau0, L0, w =", R0, td, L0, w
        pypl.subplot(2, 1, 1)
        pypl.plot(t, A82LC_full(t, y, w, tg, MNi, Ethi), color=ci)
        pypl.subplot(2, 1, 2)
        pypl.semilogy(t, A82LC_full(t, y, w, tg, MNi, Ethi), color=ci)
    pypl.show()

def test_A82LC_gp():
    """Test plots for the Gaussian process stuff"""
    # Set up a Gaussian process interpolator
    gpint = A82LC_gp("a82lcgp_4d_alt.pkl")
    t = np.arange(0.0, 120.1, 0.5)
    test_resids = True
    def my_plot_set(p, c, l):
        res = [ ]
        for pi, ci, li in zip(p, c, l):
            gpfit = gpint(t, pi)
            pypl.semilogy(t, gpfit, color=ci, label=li)
            if test_resids:
                orig = A82LC_full(t, *pi)
            else:
                orig = gpfit
            pypl.semilogy(t, orig, color=ci, ls='--')
            # calculate residuals
            res.append((orig - gpfit)/orig)
        res = np.array(res).ravel()
        res = res[abs(res) < np.inf]
        print "nmad, rms, max resids = {0:.4f}, {1:.4f}, {2:.4f};".format(
                np.median(np.abs(res)), res.std(), np.abs(res).max()),
        nok, ntot = np.sum(np.abs(res.ravel()) > 0.02), len(res.ravel())
        fok = nok / (1.0*ntot)
        print "fvals(res > 2\%) = {0}/{1} = {2:.2f}\%".format(
                nok, ntot, 100.0*fok)
        sys.stdout.flush()
        pypl.legend()
        pypl.show()
    # Vary y
    y = np.arange(0.7, 1.41, 0.05)
    pars = [(yi, 0.0, 40.0, 0.6, 0.0) for yi in y]
    colors = color_ramp(len(pars))
    labels = ["y = {0:.2f}".format(yi) for yi in y]
    print "varying y:",
    my_plot_set(pars, colors, labels)
    # Vary w with Eth0 = 0
    w = np.arange(0.0, 0.26, 0.05)
    pars = [(1.0, wi, 40.0, 0.6, 0.0) for wi in w]
    colors = color_ramp(len(pars))
    labels = ["w = {0:.2f}".format(wi) for wi in w]
    print "varying w:",
    my_plot_set(pars, colors, labels)
    # Vary w with Eth0 = 0.5e+51 erg
    w = np.arange(0.0, 0.26, 0.05)
    pars = [(1.0, wi, 40.0, 0.6, 0.5e+51) for wi in w]
    colors = color_ramp(len(pars))
    labels = ["w = {0:.2f}".format(wi) for wi in w]
    print "varying w:",
    my_plot_set(pars, colors, labels)
    # Vary tg
    tg = np.arange(20.0, 70.1, 5.0)
    pars = [(1.0, 0.0, tgi, 0.6, 0.0) for tgi in tg]
    colors = color_ramp(len(pars))
    labels = ["t$_\gamma$ = {0:.0f} days".format(tgi) for tgi in tg]
    print "varying tg:",
    my_plot_set(pars, colors, labels)

# test_A82_Lambda()
# test_A82LC_full_01()
# test_A82LC_full_02()
test_A82LC_gp()
