#!/usr/bin/env python 

"""
RS 2013/08/19:  Stritzinger 2006 MCMC implementation
"""

# ----------------------------------------------------------------------------
#                                Dependencies
# ----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as pypl
from scipy.optimize import curve_fit

from BoloMass.SNMCMC import SNMCMC
import BoloMass.Arnett82 as Arnett82
from BoloMass.Utils import BadInputError, VerboseMsg

# ----------------------------------------------------------------------------
#                       Class and function definitions
# ----------------------------------------------------------------------------

class Stritz06_MCMC(SNMCMC):
    """Stritzinger+ 2006 models for comparison with Scalzo+ 2012"""

    # Features of standard blob of chain, with upper and lower bounds
    _features = {
       # name,     def,  blo,  hlo,  hhi,  bhi,  res,  fmt, label
       'MWD':    [ 1.4,  0.0,  0.5,  2.5,  2.8, 0.05, "{0:8.2f}",
                  "Ejected mass (M$_\\odot$)"],
       'MNi':    [ 0.6,  0.0,  0.0,  2.0,  2.8, 0.05, "{0:8.2f}",
                  "$^{56}$Ni mass (M$_\\odot$)"],
       't0':     [  40,   15,   15,   75,   75,    2, "{0:8.1f}",
                  "$^{56}$Co $\\gamma$-ray timescale t$_0$(days)"],
       'kappa':  [ 0.025, 0.0, 0.02, 0.03, 0.05, 0.0005, "{0:8.3f}",
                  "$^{56}$Co $\\gamma$-ray opacity $\\kappa$ (cm$^2$ g$^{-1}$)"],
       'v_e':    [ 3000, 1000, 2000,  4000,  5000, 100, "{0:8.0f}",
                  "Scaling velocity v$_{e}$ (km s$^{-1}$)"],
       'vsc':    [10000, 7000, 7000, 14000, 14000, 100, "{0:8.0f}",
                  "Scaling velocity v$_{sc}$ (km s$^{-1}$)"],
       'muoff':  [ 0.0, -10.0, -1.0,  1.0,  10.0, 0.05, "{0:8.2f}",
                  "Distance modulus systematic (mag)"],
       'q':      [ 0.33, 0.0,  0.0,  0.65,  1.0,  0.05, "{0:8.2f}",
                  "$^{56}$Ni distribution form factor q"],
       'trise':  [17.6, 10.0, 10.0, 24.0, 24.0,  0.5, "{0:8.1f}",
                  "Rise time (days)"],
       'uvcor':  [ 0.1,  0.0,  0.0,  0.2,  0.2, 0.01, "{0:8.2f}",
                  "UV flux correction fraction"],
    }
    # Names of features to be used as main MCMC parameters
    _mcmcpars = ['MWD', 'MNi', 'v_e', 'kappa', 'q', 'trise', 'muoff', 'uvcor']
    # Names of features for which to report marginalized confidence intervals
    _confints = ['MWD', 'MNi', 't0', 'v_e', 'vsc',
                 'kappa', 'q', 'trise', 'muoff', 'uvcor']
    # Description of subplots to plot when MCMC is done running
    # In _subplots, a 1-tuple produces a histogram, while a 2-tuple produces
    # a marginalized joint confidence contour plot for both features.
    _subplot_layout = (2, 2)
    _subplots = [ ('MNi', 'MWD'), ('MWD',), ]
    _contlvls = [0.01, 0.05, 0.16, 0.50, 0.84, 0.95, 0.99]

    # Default keywords for __init__, with default values
    # These include the default Stritzinger priors on trize, kappa, v_e, and q.
    _init_kwdef = { 'muoff':  0.0, 'muoff_err': 1e-3,
                    'uvcor_lo': 0.0, 'uvcor_hi': 0.1,
                    'trise': 19.0, 'trise_err':  3.0,
                    'kappa': 0.025, 'kappa_err': 0.0025,
                    'v_e': 3000, 'v_e_err':  300,
                    'q': 0.3333, 'q_err':  0.1,
                    'sn_name': "", 'verbose': True }

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
        self.vmsg = VerboseMsg(prefix="Stritz06_MCMC({0})".format(self.name),
                               verbose=kw['verbose'], flush=True)
        # In this version we only care about stuff near maximum light and at
        # late times.  Separate out the points we're actually going to fit.
        AA = np.array
        self.tfit, self.Lfit, self.dLfit = AA(t), AA(L), AA(dL)
        t_max, t_late = np.abs(self.tfit).min(), 39.9
        idx = np.any([self.tfit == t_max, self.tfit > t_late], axis=0)
        nidx = np.invert(idx)
        self.ifit, self.infit = idx, nidx
        self.tfit, self.tnfit = self.tfit[idx], self.tfit[nidx]
        self.Lfit, self.Lnfit = self.Lfit[idx], self.Lfit[nidx]
        self.dLfit, self.dLnfit = self.dLfit[idx], self.dLfit[nidx]
        self.vmsg("Fitting the following light curve:\n",
            "\n".join(["{0:8.1f} {1:.3e} {2:.3e}".format(ti, Li, dLi)
                       for ti, Li, dLi in zip(self.tfit, self.Lfit, self.dLfit)]))
        # Priors other than hard parameter bounds
        self.trise_Pmu, self.trise_Psig = kw['trise'], kw['trise_err']
        self.muoff_Pmu, self.muoff_Psig = kw['muoff'], kw['muoff_err']
        self.kappa_Pmu, self.kappa_Psig = kw['kappa'], kw['kappa_err']
        self.v_e_Pmu, self.v_e_Psig = kw['v_e'], kw['v_e_err']
        self.q_Pmu, self.q_Psig = kw['q'], kw['q_err']
        uvcor_lo, uvcor_hi = kw['uvcor_lo'], kw['uvcor_hi']
        uvcor_av = 0.5*(uvcor_hi + uvcor_lo)
        uvcor_ss = 0.01*(uvcor_hi - uvcor_lo)
        self.uvcor_Plo, self.uvcor_Phi = uvcor_lo, uvcor_hi
        self._features['uvcor'][0:6] = [uvcor_av, uvcor_lo, uvcor_lo,
                                        uvcor_hi, uvcor_hi, uvcor_ss]
        # A chi-square fit to the light curve has at most 2 degrees of freedom
        # (MNi, t0) for goodness-of-fit, since those are the only parameters
        # we're *fitting* (rest are marginalized).
        self.ndof = max(1, len(self.tfit) - 2)
        # Quick fit to the light curve and use it as the initial guess,
        # if we have enough points; otherwise just assume the default.
        if len(self.tfit) > 2:
            MNi, t0 = self.leastsq_MNit0()
        else:
            MNi, t0 = 0.6, 40
        sf = self._features
        sf['MNi'][0] = MNi
        sf['MWD'][0] = 1.4 - MNi
        sf['trise'][0] = self.trise_Pmu
        sf['muoff'][0] = self.muoff_Pmu
        # Initialize the MCMC bits
        SNMCMC.__init__(self, verbose=kw['verbose'])

    def leastsq_MNit0(self):
        """Does a simple least-squares fit to get initial-guess (MNi, t0)"""
        # Adjust for UV correction and distance modulus errors
        Kmax = (1 + 0.5*(self.uvcor_Plo + self.uvcor_Phi))
        my_tfit = self.tfit + self.trise_Pmu
        my_Lfit = self.Lfit * 10**(-0.4*self.muoff_Pmu)
        my_dLfit = self.dLfit * 10**(-0.4*self.muoff_Pmu)
        my_Lfit[self.tfit < 10.0] *= Kmax
        my_dLfit[self.tfit < 10.0] *= Kmax
        # Fit the curve
        epsL = lambda t, MNi, t0:  MNi * Arnett82.epsilon(t, t0)
        popt, pcov = curve_fit(epsL, my_tfit, my_Lfit, sigma=my_dLfit,
                               p0=[0.5, 40])
        MNi_fit, t0_fit = popt
        MNi_err, t0_err = pcov[0,0]**0.5, pcov[1,1]**0.5
        # Report the results
        self.vmsg("least squares fit gives",
                  "MNi = {0:.2f} +/- {1:.2f} Msol".format(MNi_fit, MNi_err),
                  "t0 = {0:.1f} +/- {1:.1f} days".format(t0_fit, t0_err))
        resid = epsL(my_tfit, MNi_fit, t0_fit) - my_Lfit
        chisq = np.sum((resid/my_dLfit)**2)
        self.vmsg("chisq/nu = {0:.2f}/{1} = {2:.2f}".format(
                  chisq, self.ndof, chisq/self.ndof))
        return MNi_fit, t0_fit

    def fillblob(self, pars):
        """Fills a blob with everything but likelihood and prior

        Since PTSampler doesn't support blobs, it makes sense for us to break
        out the blob-filling capabilities so that we don't do them over and
        over needlessly.  In point of fact, we need to calculate most of the
        blob quantities to calculate the likelihood, but there's no sense in
        re-evaluating the log likelihood if all we want is the blob.
        """
    
        # Unpack parameters from vector
        MWD, MNi, v_e, kappa, q, trise, muoff, uvcor = pars
        # Default blob
        blob = dict(self._default_blob)
        blob.update({ 'MWD': MWD, 'MNi': MNi, 'v_e': v_e, 'q': q,
                      'trise': trise, 'muoff': muoff, 'uvcor': uvcor,
                      'fail': 'default' })
        # Get rid of unphysical fits with extreme prejudice
        if MWD < MNi:
            blob['fail'] = "badMWD"
            return blob
        # Fill the regular vsc for comparison with other models
        vsc = np.sqrt(12)*v_e
        # Calculate t0 based on v_e from Stritzinger's original expression
        t0 = np.sqrt((MWD*2e+33) * kappa * q / (8*np.pi)) / (v_e*1e+5) / 86400
        # Update blob with physical solution
        blob.update({ 'vsc': vsc, 'v_e': v_e, 't0': t0, 'q': q, 'fail': None })
        return blob

    def logl(self, pars, blob=None):
        """Log likelihood *only* for PTSampler

        This assumes that all the parameters lie within bounds laid out in
        self._features.  Implicit bounds caused by physics *necessary* for the
        likelihood to make sense, e.g., binding energy, must be included here.
        """

        # Fill the blob first, if necessary
        if blob is None:
            blob = Stritz06_MCMC.fillblob(self, pars)
        if blob['fail'] is not None:
            return -np.inf
        MNi, trise, t0, muoff, uvcor = [blob[f] for f in
            ('MNi', 'trise', 't0', 'muoff', 'uvcor')]
        # Model:  energy deposited (effective alpha = 1)
        model = MNi * Arnett82.epsilon(self.tfit + trise, t0)
        # Data:  include distance modulus offset, plus UV near max
        data = self.Lfit * 10**(-0.4*muoff)
        data[self.tfit < 10.0] *= (1 + uvcor) 
        # Calculate chi-square
        chisq = (((model - data)/self.dLfit)**2).sum()
        return self.lnPchisq(chisq, self.ndof)

    def logp(self, pars, blob=None):
        """Log prior *only* for PTSampler

        This assumes that all the parameters lie within bounds laid out in
        self._features.  Implicit bounds caused by physics assumed beyond
        what's needed to calculate the likelihood, e.g., neutronization,
        must be included here.
        """

        # Unpack parameters from vector
        MWD, MNi, v_e, kappa, q, trise, muoff, uvcor = pars
        # Prior terms P(theta), some fixed in the Stritzinger+ 2006 text
        chpri = ((trise - self.trise_Pmu) / self.trise_Psig)**2
        chpri += ((muoff - self.muoff_Pmu) / self.muoff_Psig)**2
        chpri += ((kappa - self.kappa_Pmu) / self.kappa_Psig)**2
        chpri += ((v_e - self.v_e_Pmu) / self.v_e_Psig)**2
        chpri += ((q - self.q_Pmu) / self.q_Psig)**2
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
        SChidx = np.array([b['MWD'] > 1.4 for b in goodblobs])
        SChblobs = goodblobs[SChidx]
        SChprobs = np.exp(goodprobs[SChidx])
        print "   fraction of samples with MWD > 1.4:  {0}".format(
                len(SChblobs) / float(len(goodblobs)))
        if len(SChblobs) > 0:
            print "   highest probability with MWD > 1.4:  {0}".format(
                    np.max(SChprobs))
        # Calculate the best-fit light curve
        best_blob = goodblobs[goodprobs.argmax()]
        self.best_model_t = np.arange(
                -5.0, max(self.tfit[-1], self.tnfit[-1], 1.0))
        self.best_model_L = best_blob['MNi'] * Arnett82.epsilon(
                self.best_model_t + best_blob['trise'], best_blob['t0'])

        # Then show the light curve fit in the bottommost panel
        if not makeplots:  return
        pypl.subplot(2, 1, 2)
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
        if plotfname:
            pypl.savefig(plotfname, dpi=100)
        if showplots:
            pypl.show()
