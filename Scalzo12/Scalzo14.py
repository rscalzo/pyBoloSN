#!/usr/bin/env python 

"""
RS 2013/08/24:  Rewriting SN Ia Monte Carlo Markov Chain Reconstruction

This module (Scalzo14) is basically the same as Scalzo12, except it tries to
fit the full Arnett model to as much of the light curve as it can manage.
I'm still getting rid of the section 10 < t < 30 to be safe, though honestly,
"""

# ----------------------------------------------------------------------------
#                                Dependencies
# ----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as pypl
from scipy.optimize import curve_fit, brent, brentq, fmin_tnc

from BoloMass.SNMCMC import SNMCMC
import BoloMass.SNPhysics as SNPhysics
import BoloMass.Arnett82 as Arnett82
import BoloMass.Jeffery99 as Jeffery99
from BoloMass.Utils import BadInputError, VerboseMsg

# ----------------------------------------------------------------------------
#                       Class and function definitions
# ----------------------------------------------------------------------------

class Scalzo14_MCMC(SNMCMC):
    """Scalzo+ 2012 SN Ia models plus an Arnett light curve fit"""

    # Features of standard blob of chain, with upper and lower bounds
    _features = {
       # name,     def,  blo,  hlo,  hhi,  bhi,  res,  fmt, label
       'MWD':    [ 1.4,  0.8,  0.5,  2.5,  2.8, 0.05, "{0:8.2f}",
                  "Ejected mass (M$_\\odot$)"],
       'MFe':    [ 0.1,  0.0,  0.0,  0.5,  2.0, 0.01, "{0:8.2f}",
                  "Stable Fe-peak mass (M$_\\odot$)"],
       'MNi':    [ 0.6,  0.0,  0.0,  2.0,  2.8, 0.05, "{0:8.2f}",
                  "$^{56}$Ni mass (M$_\\odot$)"],
       'MSi':    [ 0.6,  0.0,  0.0,  2.8,  2.8, 0.05, "{0:8.2f}",
                  "$IME (S, Si, Mg, Ca) mass (M$_\\odot$)"],
       'MCO':    [ 0.1,  0.0,  0.0,  0.5,  2.8, 0.01, "{0:8.2f}",
                  "Unburned C+O mass (M$_\\odot$)"],
       'y':      [ 1.0,  0.6,  0.7,  1.4,  1.6, 0.05, "{0:8.2f}",
                  "Arnett light curve width parameter"],
       'w':      [ 0.01, 0.0,  0.0,  0.02, 0.02, 0.001, "{0:8.3f}",
                  "Arnett finite-size parameter"],
       't0':     [  40,   15,   15,   75,   75,    2, "{0:8.1f}",
                  "$^{56}$Co $\\gamma$-ray timescale t$_0$(days)"],
       'vsc':    [10000, 7000, 7000, 14000, 14000, 100, "{0:8.0f}",
                  "Scaling velocity v$_{sc}$ (km s$^{-1}$)"],
       'muoff':  [ 0.0, -10.0, -1.0,  1.0,  10.0, 0.05, "{0:8.2f}",
                  "Distance modulus systematic (mag)"],
       'lpc':    [ 9.0,  7.0,  7.0, 10.0,  9.7,  0.1, "{0:8.2f}",
                  "log$_{10}$(Central density in g cm$^{-3}$)"],
       'Q':      [ 3.0,  0.0,  0.0,  5.0,  5.0,  0.1, "{0:8.2f}",
                  "$^{56}$Ni distribution form factor Q"],
       'alpha':  [ 1.2,  0.8,  0.8,  1.6,  1.6, 0.05, "{0:8.2f}",
                  "Radiation trapping form factor $\\alpha$"],
       'trise':  [17.6, 10.0, 10.0, 24.0, 24.0,  0.5, "{0:8.1f}",
                  "Rise time (days)"],
       'texp':   [-17.6, -24.0, -24.0, -10.0, -10.0, 0.5, "{0:8.1f}",
                  "Phase of explosion date (days)"],
       'aNi':    [ 0.2,  0.0,  0.0,  0.5,  0.5, 0.02, "{0:8.2f}",
                  "$^{56}$Ni mixing scale"],
       'Eth51':  [ 0.1,  0.0,  0.0,  2.0,  2.0,  0.05, "{0:8.2f}",
                  "Initial thermal energy (10$^{51}$ erg)"],
       'Egrv51': [ 1.0,  0.0,  0.0,  2.0,  2.0,  0.1, "{0:8.2f}",
                  "Gravitational binding energy (10$^{51}$ erg)"],
       'Ekin51': [ 1.0,  0.0,  0.0,  2.0,  2.0,  0.1, "{0:8.2f}",
                  "Kinetic energy (10$^{51}$ erg)"],
    }
    # Names of features to be used as main MCMC parameters
    # Note that by default (i.e. for normal Ia) Eth51 and w are fixed to zero.
    _mcmcpars = ['MFe', 'MNi', 'MSi', 'MCO', 'lpc',
                 'y', 'w', 'Eth51', 'muoff', 'texp', 'aNi']
    # Names of features for which to report marginalized confidence intervals
    _confints = ['MWD', 'MNi', 'MFe', 'MSi', 'MCO', 'trise', 'y', 'w',
                 't0', 'vsc', 'muoff', 'lpc', 'Q', 'alpha',
                 'Eth51', 'Egrv51', 'Ekin51', 'aNi']
    # Description of subplots to plot when MCMC is done running
    # In _subplots, a 1-tuple produces a histogram, while a 2-tuple produces
    # a marginalized joint confidence contour plot for both features.
    _subplot_layout = (3, 2)
    _subplots = [ ('MNi', 'MWD'), ('Eth51', 'w'), ('MWD',), ('MCO', 'vsc') ]
    _contlvls = [0.01, 0.05, 0.16, 0.50, 0.84, 0.95, 0.99]

    # Default keywords for __init__, with default values
    _init_kwdef = { 'muoff':    0.0, 'muoff_err': 1e-3,
                    'eta':      0.9, 'eta_err':    0.1,
                    'trise':   17.6, 'trise_err':  2.0,
                    'uvcor_lo': 0.0, 'uvcor_hi':   0.1,
                    'aNi':      0.2, 'aNi_err':   0.02,
                    'wgt0':    True, 'qtabpkl':   None,
                    'holefe':  True,
                    'sn_name':   "", 'verbose':   True }

    def __init__(self, t, L, dL, **init_kwargs):
        """Initialize!
        
           t:  time since bolometric maximum light, in days
           L:  bolometric luminosity in erg/s
           dL:  1-sigma error on bolometric luminosity in erg/s
           muoff:  offset to apply to distance modulus, if any
           muerr:  error on distance modulus offset
           qtabpkl:  filename of pickled QGP object
           sn_name:  name of SN (optional)
        """
        # Unpack kwargs
        kw = dict(self._init_kwdef)
        kw.update(init_kwargs)
        # First-pass initialization
        self.name = kw['sn_name']
        self.vmsg = VerboseMsg(prefix="Scalzo14_MCMC({0})".format(self.name),
                               verbose=kw['verbose'], flush=True)
        self.qgp = Jeffery99.QGP(kw['qtabpkl'])
        self.a82gp = Arnett82.stdLCgp
        # In this version we only care about stuff near maximum light and at
        # late times.  Separate out the points we're actually going to fit.
        AA = np.array
        self.t, self.L, self.dL = AA(t), AA(L), AA(dL)
        t_max, t_early, t_late = np.abs(self.t).min(), 10.0, 39.9
        idx = np.array([False for t in self.t])
        idx[np.any([self.t < t_early, self.t > t_late], axis=0)] = True
        idx[self.t > 100] = False
        idx[abs(self.t) < 0.01] = False
        nidx = np.invert(idx)
        self.ifit, self.infit = idx, nidx
        # Add the UV correction in as an iid systematic dispersion to the
        # data near max.  In a better world we'd try to model it, maybe with
        # a curve drawn from a GP posterior; I may try it in the next version.
        uvcor = 0.5*(kw['uvcor_hi'] + kw['uvcor_lo'])
        uvdisp = 0.3*(kw['uvcor_hi'] - kw['uvcor_lo'])
        self.dL[self.t < 10.0] = np.sqrt(
            ((1 + uvcor) * self.dL[self.t < t_early])**2 +
            (uvdisp * self.L[self.t < t_early])**2)
        self.L[self.t < t_early] *= (1 + uvcor)
        self.tfit, self.tnfit = self.t[idx], self.t[nidx]
        self.Lfit, self.Lnfit = self.L[idx], self.L[nidx]
        self.dLfit, self.dLnfit = self.dL[idx], self.dL[nidx]
        # Make sure we can actually do this!
        if len(self.tfit) < 4:
            raise BadInputError("not enough data points to use Arnett fit")
        if len(self.tfit[self.tfit < 0]) < 2:
            raise BadInputError("early-time Arnett fit poorly constrained")
        # Hard bounds on initial radius if we want them
        self.wgt0 = kw['wgt0']
        if not self.wgt0:
            self._features['w'][0:6] = [0.5e-7, 0.0, 0.0, 1e-7, 1e-7, 1e-9]
            self._features['Eth51'][0:6] = [0.5e-8, 0.0, 0.0, 1e-8, 1e-8, 1e-9]
        # Priors other than hard parameter bounds
        self.holefe = kw['holefe']
        self.eta_Pmu, self.eta_Psig = kw['eta'], kw['eta_err']
        self.aNi_Pmu, self.aNi_Psig = kw['aNi'], kw['aNi_err']
        self.trise_Pmu, self.trise_Psig = kw['trise'], kw['trise_err']
        self.muoff_Pmu, self.muoff_Psig = kw['muoff'], kw['muoff_err']
        # Number of degrees of freedom for the chi-square fit.  We're varying
        # MNi, t0, and y; w and Eth51 are really just nuisance parameters.
        self.ndof = max(1, len(self.tfit) - 3)
        # Initialize the MCMC bits
        SNMCMC.__init__(self, verbose=kw['verbose'])
        # Quick fit to the light curve and use it as the initial guess,
        # if we have enough points; otherwise just assume the default.
        self.vmsg("Fitting the following light curve:\n",
            "\n".join(["{0:8.1f} {1:.3e} {2:.3e}".format(ti, Li, dLi)
                       for ti, Li, dLi in zip(self.tfit, self.Lfit, self.dLfit)]))
        p0 = self.leastsq_arnett()
        # Set the initial guess ball
        for ki, pi in zip(self._mcmcpars, p0):  self._features[ki][0] = pi
        SNMCMC.__init__(self, verbose=kw['verbose'])

    def leastsq_arnett(self):
        """Does a simple least-squares fit to get initial-guess (MNi, t0)"""
        # Adjust for UV correction and distance modulus errors
        my_Lfit = self.Lfit * 10**(-0.4*self.muoff_Pmu)
        my_dLfit = self.dLfit * 10**(-0.4*self.muoff_Pmu)
        my_Lnfit = self.Lnfit * 10**(-0.4*self.muoff_Pmu)
        my_dLnfit = self.dLnfit * 10**(-0.4*self.muoff_Pmu)
        if self.wgt0:
            # Use the full Arnett solution accounting for nonzero initial size
            pnames = ('y', 'MNi', 't0', 'w', 'Eth51', 'texp')
            def epsL(t, y, MNi, t0, w, Eth, texp):
                L_early = self.a82gp(t[t < 100]-texp, (y, w, t0, MNi, Eth*1e+51))
                L_late = Arnett82.A82LC_full(t[t > 100]-texp, y, w, t0, MNi, Eth*1e+51)
                return np.concatenate([L_early, L_late])
            p0 = [1.0, 0.5, 40.0, 0.01, 0.5, -17.6]
            popt, pcov = curve_fit(epsL, self.tfit, my_Lfit, sigma=my_dLfit, p0=p0)
            # Make sure w isn't negative!
            if popt[3] < 0.0:
                popt[3] *= -1
                pcov[:,3] *= -1
                pcov[3,:] *= -1
        else:
            # Just use the zero-initial-size Arnett solution
            pnames = ('y', 'MNi', 't0', 'w', 'Eth51', 'texp')
            def epsL(t, y, MNi, t0, texp):
                L_early = self.a82gp(t[t < 100]-texp, (y, 0.0, t0, MNi, 0.0))
                L_late = Arnett82.A82LC_full(t[t > 100]-texp, y, 0.0, t0, MNi, 0.0)
                return np.concatenate([L_early, L_late])
            p0 = [1.0, 0.5, 40.0, -17.6]
            popt, pcov = curve_fit(epsL, self.tfit, my_Lfit, sigma=my_dLfit, p0=p0)
        # Now carry on
        pcovI = np.matrix(pcov).I
        perr = np.diag(pcov)**0.5
        self.lc_pnames, self.lc_popt, self.lc_pcovI = pnames, popt, pcovI
        # Report the results
        self.vmsg("least squares fit gives:\n",
                  "".join(["  {0:5s} = {1:8.3f} +/- {2:5.3f}\n".format
                  (pn, pi, pe) for pn, pi, pe in zip(pnames, popt, perr)]))
        # Now fix all the light curve parameters to their optimal values.
        if self.wgt0:
            y_fix, MNi_fix, t0_fix, w_fix, Eth_fix, texp_fix = popt
        else:
            y_fix, MNi_fix, t0_fix, texp_fix = popt
            w_fix, Eth_fix = 0.0, 0.0
        # Compute the actual rise time from the explosion time.
        def MepsL(ti):
            return -Arnett82.A82LC_full(
                    ti, y_fix, w_fix, t0_fix, MNi_fix, Eth_fix*1e+51)
        trise = brent(MepsL, brack=(5, 17.6*y_fix, 40))
        # Calculate alpha while we're at it
        Lmax = -MepsL(trise)
        emax = MNi_fix * Arnett82.epsilon(trise, t0_fix)
        alpha = Lmax/emax
        self.vmsg("trise =", trise, "alpha =", alpha)
        # Calculate chi-square of fit
        model, data, errs = epsL(self.tfit, *popt), self.Lfit, self.dLfit
        chisq = (((model-data)/errs)**2).sum()
        self.vmsg("chisq/ndof = {0}/{1} = {2}".format(
                  chisq, self.ndof, chisq/self.ndof))
        self.vmsg("lnprob =", self.lnPchisq(chisq, self.ndof))
        # Find plausible initial guesses for all the other parameters.
        #    MNi, y, w, Eth51, texp:  from fit
        #    muoff, aNi:  strong priors
        #    MFe, MCO:  plausible guesses easy
        #    MSi, lpc:  vary to reproduce best light curve
        # First off:  since we're using the Yoon & Langer 2005 binding energy
        # formula, more massive SNe Ia can always have lower central density.
        # So lowball the central density.  For this choice, MFe will be small.
        muoff, aNi = self.muoff_Pmu, self.aNi_Pmu
        # Solutions should exist with small MFe, MCO.
        MFe, MCO = 0.05*MNi_fix, 0.05*MNi_fix
        # Now vary MSi and lpc to fit.
        aNi, MSi, lpc = 0.1, 0.7, 8.5
        def tresid(myMSi):
            # generalized q lookup, (aNi, MFe, MNi) => (aNi, mNilo, mNihi)
            # for treatment of 56Ni hole see Scalzo12_MCMC.fillblob
            MWD = MFe + MNi_fix + myMSi + MCO
            if self.holefe:
                raNi = 1.5*aNi
                if MFe/MWD > raNi:
                    mNilo = MFe/MWD
                else:
                    mNilo = np.sqrt(4*raNi*MFe/MWD) - raNi
                mNihi = (MFe + MNi_fix)/MWD
            else:
                mNilo, mNihi = -0.5, (MFe + MNi_fix)/MWD
            q = self.qgp.eval([mNilo, mNihi, 0.7*aNi])
            t0 = Jeffery99.t0(q, MWD*2e+33, MWD*1.5e+51)
            return (t0 - t0_fix)**2
        MSi = float(brent(tresid, tol=1e-3, brack=(0, 0.7, 2.0)))
        MWD = MFe + MNi_fix + MSi + MCO
        lpc = 2.5*(MWD - 0.8) + 7.0
        print "Picking MSi = {0:.2f} Msol, lpc = {1:.1f}".format(MSi, lpc)
        chainpars = [MFe, MNi_fix, MSi, MCO, lpc,
                     y_fix, w_fix, Eth_fix, muoff, texp_fix, aNi]
        # Plot best fit?
        plot = False
        if plot:
            tmod = np.arange(texp_fix, max(self.tfit[-1], self.tnfit[-1]), 1.0)
            Lmod = self.a82gp(tmod - texp_fix,
                              (y_fix, w_fix, t0_fix, MNi_fix, Eth_fix*1e+51))
            emod = MNi_fix * Arnett82.epsilon(tmod - texp_fix, t0_fix)
            pypl.semilogy(tmod, Lmod)
            pypl.plot(tmod, emod)
            pypl.errorbar(self.tfit, my_Lfit, yerr=my_dLfit,
                          ls='None', marker='o', c='g')
            pypl.errorbar(self.tnfit, my_Lnfit, yerr=my_dLnfit,
                          ls='None', marker='o', c='r')
            pypl.show()
        print "Initial guess parameters:", chainpars
        return chainpars

    def fillblob(self, pars):
        """Fills a blob with everything but likelihood and prior

        Since PTSampler doesn't support blobs, it makes sense for us to break
        out the blob-filling capabilities so that we don't do them over and
        over needlessly.  In point of fact, we need to calculate most of the
        blob quantities to calculate the likelihood, but there's no sense in
        re-evaluating the log likelihood if all we want is the blob.
        """

        # Unpack parameters from vector
        MFe, MNi, MSi, MCO, lpc, y, w, Eth51, muoff, texp, aNi = pars
        # Total white dwarf mass
        MWD = MFe + MNi + MSi + MCO
        # Default blob
        blob = dict(self._default_blob)
        blob.update({ 'MFe': MFe, 'MNi': MNi, 'MSi': MSi, 'MCO': MCO,
            'MWD': MWD, 'lpc': lpc, 'y': y, 'w': w, 'muoff': muoff,
            'Eth51': Eth51, 'aNi': aNi, 'texp': texp, 'trise': -17.6*y,
            'fail': 'default' })
        # Energetics -- all energies in ergs
        try:
            Egrv = SNPhysics.yl05_bind(MWD, lpc)
        except BadInputError:
            Egrv = -np.inf
        Eth = Eth51 * 1e+51
        Enuc = SNPhysics.mi09_Enuc(MFe, MNi, MSi, MCO)
        Ekin = Enuc - Egrv # - Eth
        Enuc51, Egrv51, Ekin51 = Enuc/1e+51, Egrv/1e+51, Ekin/1e+51
        blob.update({ 'Egrv51': Egrv51, 'Enuc51': Enuc51, 'Ekin51': Ekin51 })
        # Get rid of unphysical fits with extreme prejudice
        if Egrv < 0 or Egrv != Egrv:
            blob['fail'] = "badEgrv"
            return blob
        elif Ekin < 0 or Ekin != Ekin:
            blob['fail'] = "badEkin"
            return blob
        elif Eth < 0 or Eth > Ekin or Eth != Eth:
            blob['fail'] = "badEth"
            return blob
        elif texp > self.tfit[-1]:
            blob['fail'] = "badtexp"
            return blob
        # scaling velocity of the ejecta
        vsc = SNPhysics.Density.v_sc(MWD*2e+33, Ekin)
        # generalized q lookup, (aNi, MFe, MNi) => (aNi, mNilo, mNihi)
        # assuming no central 56Ni hole
        if self.holefe:
            raNi = 1.5*aNi
            if MFe/MWD > raNi:
                mNilo = MFe/MWD
            else:
                mNilo = np.sqrt(np.abs(4*raNi*MFe/MWD)) - raNi
            mNihi = (MFe + MNi)/MWD
        else:
            mNilo, mNihi = -0.5, (MFe + MNi)/MWD
        q = self.qgp.eval([mNilo, mNihi, 0.7*aNi])
        t0 = Jeffery99.t0(q, MWD*2e+33, Ekin)
        """
        # Find the actual rise time by isolating the maximum
        brack = (-7, 0, 7)
        def MepsL(t):
            return -self.a82gp(t - texp, (y, w, t0, MNi, Eth51*1e+51))
        try:
            tmax, Lmax, niter, ncalls = brent(
                    MepsL, tol=1e-3, brack=brack, full_output=True)
            trise = tmax - texp
            Lmax *= -1
        except:
            trise, Lmax = -texp, -MepsL(0)
        emax = MNi * Arnett82.epsilon(trise, t0)
        alpha = Lmax/emax
        # Update blob with physical solution
        """
        alpha, trise = 1.0, -texp
        blob.update({ 'fail': None, 'vsc': vsc/1e+5, 't0': t0,
                      'Q': q, 'alpha': alpha, 'trise': trise })
        return blob

    def logl(self, pars, blob=None):
        """Log likelihood *only* for PTSampler

        This assumes that all the parameters lie within bounds laid out in
        self._features.  Implicit bounds caused by physics *necessary* for the
        likelihood to make sense, e.g., binding energy, must be included here.
        """

        # Fill the blob first, if necessary
        if blob is None:
            blob = Scalzo14_MCMC.fillblob(self, pars)
        if blob['fail'] is not None:
            return -np.inf
        MNi, Eth51, texp, trise, y, w, t0, muoff = [blob[f] for f in
            ('MNi', 'Eth51', 'texp', 'trise', 'y', 'w', 't0', 'muoff')]
        if False and self.lc_pcovI is not None:
            # If we had a good Arnett fit, just use the covariance matrix,
            # accounting for the distance modulus error
            MNip = MNi * 10**(-0.4*muoff)
            Eth51p = Eth51 * 10**(-0.4*muoff)
            p = np.array([y, MNip, t0, w, Eth51p, texp])
            dp = p - self.lc_popt
            chisq = np.dot(dp, np.dot(np.asarray(self.lc_pcovI), dp))
            # print "pnames = [y, MNi, t0, w, Eth51, texp]"
            # print "p = [{0:.2f} {1:.2f} {2:.1f} {3:.3f} {4:.2f} {5:.1f}]".format(*p)
            # print "dp = [{0:.2f} {1:.2f} {2:.1f} {3:.3f} {4:.2f} {5:.1f}]".format(*dp)
            # print "cov chisq =", chisq
        else:
            # Model:  energy deposited
            model = self.a82gp(self.tfit - texp, (y, w, t0, MNi, Eth51*1e+51))
            # Data:  include distance modulus offset, plus UV near max
            data = self.Lfit * 10**(-0.4*muoff)
            # Calculate chi-square
            chisq = (((model - data)/self.dLfit)**2).sum()
            # print "fit chisq =", chisq
        # A bit of a hack:  Include this prior term in there so that we don't
        # have to recalculate the blob in logp
        chpri = ((trise - self.trise_Pmu) / self.trise_Psig)**2
        return self.lnPchisq(chisq, self.ndof) - 0.5*chpri

    def logp(self, pars, blob=None):
        """Log prior *only* for PTSampler

        This assumes that all the parameters lie within bounds laid out in
        self._features.  Implicit bounds caused by physics assumed beyond
        what's needed to calculate the likelihood, e.g., neutronization,
        must be included here.
        """

        # Unpack parameters from vector
        MFe, MNi, MSi, MCO, lpc, y, w, Eth51, muoff, texp, aNi = pars
        # Total white dwarf mass
        MWD = MFe + MNi + MSi + MCO
        # Neutronization prior (Krueger+ 2012, Seitenzahl+ 2013)
        eta = MNi/(MFe + MNi)
        self.eta_Pmu, self.eta_Psig = SNPhysics.krueger12_eta(lpc)
        # Other prior terms P(theta) for ancillary parameters
        chpri = ((muoff - self.muoff_Pmu) / self.muoff_Psig)**2
        chpri += ((eta - self.eta_Pmu) / self.eta_Psig)**2
        chpri += ((aNi - self.aNi_Pmu) / self.aNi_Psig)**2
        chpri += ((MCO/MWD - 0.0) / 0.05)**2
        return -0.5*chpri

    def show_results(self, makeplots=True, showplots=True, plotfname=None):
        """Overload of show_results including light curve fit"""

        # If we're going to save or show plots, we have to make them first
        if plotfname or showplots:  makeplots = True

        # First show the contour plots etc.
        SNMCMC.show_results(self, makeplots=makeplots, showplots=False)
        if sum(self.goodidx) < 5:
            self.vmsg("No good blobs, hence no results to show!")
            return
        # Count the super-Chandra fraction etc.
        goodblobs = np.array([b for b in self.bloblist[self.goodidx]])
        goodprobs = np.array([b for b in self.lnproblist[self.goodidx]])
        SChidx = np.array([b['lpc'] > 9.0 for b in goodblobs])
        SChblobs = goodblobs[SChidx]
        SChprobs = np.exp(goodprobs[SChidx])
        print "   fraction of samples with lpc > 9.0:  {0}".format(
                len(SChblobs) / float(len(goodblobs)))
        if len(SChblobs) > 0:
            print "   highest probability with lpc > 9.0:  {0}".format(
                    np.max(SChprobs))
        # Calculate the best-fit light curve
        best_blob = goodblobs[goodprobs.argmax()]
        self.best_model_t = np.arange(
                -5.0, max(self.tfit[-1], self.tnfit[-1], 1.0))
        self.best_model_L = best_blob['MNi'] * Arnett82.epsilon(
                self.best_model_t + best_blob['trise'], best_blob['t0'])

        # Then show the light curve fit in the bottommost panel
        if not makeplots:  return
        pypl.subplot(3, 1, 3)
        pypl.plot(self.best_model_t, np.log10(self.best_model_L),
                  color='g', ls='-')
        pypl.errorbar(self.tfit, np.log10(self.Lfit),
                      yerr=np.log10(1.0 +self.dLfit/self.Lfit),
                      c = 'g', ls='None', marker='o')
        pypl.errorbar(self.tnfit, np.log10(self.Lnfit),
                      yerr=np.log10(1.0 +self.dLnfit/self.Lnfit),
                      c = 'r', ls='None', marker='o')
        pypl.xlabel("Days Since Bolometric Maximum Light")
        pypl.ylabel("Bolometric Luminosity (erg/s)")
        fig = pypl.gcf()
        fig.set_size_inches(7.5, 7.5)
        pypl.subplots_adjust(left=0.1, right=0.9, bottom=0.10, top=0.95,
                             wspace=0.30, hspace=0.25)
        if showplots:
            pypl.show()
        if plotfname:
            pypl.savefig(plotfname)
