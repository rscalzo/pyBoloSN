#!/usr/bin/env python 

# ============================================================================
# RS 2013/01/24:  Rewriting SN Ia Monte Carlo Markov Chain Reconstruction
# ----------------------------------------------------------------------------
# Now rewriting stuff I wrote in C++ some time ago, in the hope of maybe
# releasing it as a public useful code.
# ============================================================================


# ----------------------------------------------------------------------------
#                                Dependencies
# ----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as pypl
from scipy.optimize import curve_fit

from BoloMass.SNMCMC import SNMCMC
import BoloMass.SNPhysics as SNPhysics
import BoloMass.Arnett82 as Arnett82
import BoloMass.Jeffery99 as Jeffery99
from BoloMass.Utils import BadInputError, VerboseMsg

# ----------------------------------------------------------------------------
#                       Class and function definitions
# ----------------------------------------------------------------------------

class Scalzo12_MCMC(SNMCMC):
    """Refactored representation of the Scalzo+ 2012 SN Ia models"""

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
       't0':     [  40,   15,   15,   75,   75,    2, "{0:8.1f}",
                  "$^{56}$Co $\\gamma$-ray timescale t$_0$(days)"],
       'vsc':    [10000, 7000, 7000, 14000, 14000, 100, "{0:8.0f}",
                  "KE velocity v$_{KE}$ (km s$^{-1}$)"],
       'muoff':  [ 0.0, -10.0, -1.0,  1.0,  10.0, 0.05, "{0:8.2f}",
                  "Distance offset $\Delta \mu$ (mag)"],
       'lpc':    [ 9.0,  7.0,  7.0, 10.0,  9.7,  0.1, "{0:8.2f}",
                  "log$_{10}$(Central density in g cm$^{-3}$)"],
       'Q':      [ 3.0,  0.0,  0.0,  5.0,  5.0,  0.1, "{0:8.2f}",
                  "$^{56}$Ni depth form factor Q"],
       'alpha':  [ 1.2,  0.8,  0.8,  1.6,  1.6, 0.05, "{0:8.2f}",
                  "Arnett's rule form factor $\\alpha$"],
       'trise':  [17.6, 10.0, 10.0, 24.0, 24.0,  0.5, "{0:8.1f}",
                  "Rise time (days)"],
       'uvcor':  [ 0.1,  0.0,  0.0,  0.2,  0.2, 0.01, "{0:8.2f}",
                  "Near-max UV flux correction"],
       'aNi':    [ 0.2,  0.0,  0.0,  0.5,  0.5, 0.02, "{0:8.2f}",
                  "$^{56}$Ni mixing scale"],
       'Egrv51': [ 1.0,  0.0,  0.0,  2.0,  2.0,  0.1, "{0:8.2f}",
                  "Binding energy (10$^{51}$ erg)"],
       'Ekin51': [ 1.0,  0.0,  0.0,  2.0,  2.0,  0.1, "{0:8.2f}",
                  "Kinetic energy (10$^{51}$ erg)"],
    }
    # Names of features to be used as main MCMC parameters
    _mcmcpars = ['MFe', 'MNi', 'MSi', 'MCO', 'lpc', 'alpha', 'muoff',
                 'trise', 'uvcor', 'aNi']
    # Names of features for which to report marginalized confidence intervals
    _confints = ['MWD', 'MNi', 'MFe', 'MSi', 'MCO', 'trise', 't0', 'vsc',
                 'muoff', 'lpc', 'Q', 'alpha', 'Egrv51', 'Ekin51', 'uvcor', 'aNi']
    # Description of subplots to plot when MCMC is done running
    # In _subplots, a 1-tuple produces a histogram, while a 2-tuple produces
    # a marginalized joint confidence contour plot for both features.
    _subplot_layout = (3, 2)
    # _subplots = [ ('MNi', 'MWD'), ('lpc', 'Q'), ('MWD',), ('MCO', 'vsc') ]
    _subplots = [ ('MNi', 'MWD'), ('aNi', 'MWD'), ('MWD',), ('alpha', 'MWD') ]
    _contlvls = [0.01, 0.05, 0.16, 0.50, 0.84, 0.95, 0.99]

    # Default keywords for __init__, with default values
    _init_kwdef = { 'muoff':    0.0, 'muoff_err': 1e-3,
                    'eta':      0.9, 'eta_err':    0.1,
                    'alpha':    1.2, 'alpha_err':  0.2,
                    'trise':   17.6, 'trise_err':  2.0,
                    'uvcor_lo': 0.0, 'uvcor_hi':   0.2,
                    'aNi':      0.2, 'aNi_err':    0.1,
                    'fCO':     0.00, 'fCO_err':   0.05,
                    'qtabpkl': None, 'holefe':   False,
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
        self.vmsg = VerboseMsg(prefix="Scalzo12_MCMC({0})".format(self.name),
                               verbose=kw['verbose'], flush=True)
        self.qgp = Jeffery99.QGP(kw['qtabpkl'])
        # In this version we only care about stuff near maximum light and at
        # late times.  Separate out the points we're actually going to fit.
        AA = np.array
        self.tfit, self.Lfit, self.dLfit = AA(t), AA(L), AA(dL)
        t_max, t_late = np.abs(self.tfit).min(), 39.9
        idx = np.any([np.abs(self.tfit) == t_max, self.tfit > t_late], axis=0)
        # RS 2013/09/20:  HACK to use only data with 40 < t < 45
        # idx[self.tfit > 45.0] = False
        # RS 2014/10/03:  remove all points with really crappy S/N
        idx[self.Lfit/self.dLfit < 5.0] = False
        nidx = np.invert(idx)
        self.ifit, self.infit = idx, nidx
        self.tfit, self.tnfit = self.tfit[idx], self.tfit[nidx]
        self.Lfit, self.Lnfit = self.Lfit[idx], self.Lfit[nidx]
        self.dLfit, self.dLnfit = self.dLfit[idx], self.dLfit[nidx]
        self.vmsg("Fitting the following light curve:\n",
            "\n".join(["{0:8.1f} {1:.3e} {2:.3e}".format(ti, Li, dLi)
                       for ti, Li, dLi in zip(self.tfit, self.Lfit, self.dLfit)]))
        # Priors other than hard parameter bounds
        self.holefe = kw['holefe']
        self.eta_Pmu, self.eta_Psig = kw['eta'], kw['eta_err']
        self.aNi_Pmu, self.aNi_Psig = kw['aNi'], kw['aNi_err']
        self.fCO_Pmu, self.fCO_Psig = kw['fCO'], kw['fCO_err']
        self.alpha_Pmu, self.alpha_Psig = kw['alpha'], kw['alpha_err']
        self.trise_Pmu, self.trise_Psig = kw['trise'], kw['trise_err']
        self.muoff_Pmu, self.muoff_Psig = kw['muoff'], kw['muoff_err']
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
        sf['MFe'][0] = 0.1*MNi
        sf['MSi'][0] = max(0.1, 1.3 - 1.1*MNi)
        sf['MCO'][0] = 0.1
        sf['lpc'][0] = 9.0
        sf['aNi'][0] = self.aNi_Pmu
        sf['alpha'][0] = self.alpha_Pmu
        sf['trise'][0] = self.trise_Pmu
        sf['muoff'][0] = self.muoff_Pmu
        # Initialize the MCMC bits
        SNMCMC.__init__(self, verbose=kw['verbose'])

    def leastsq_MNit0(self):
        """Does a simple least-squares fit to get initial-guess (MNi, t0)"""
        # Adjust for UV correction and distance modulus errors
        Kmax = (1 + 0.5*(self.uvcor_Plo + self.uvcor_Phi)) / self.alpha_Pmu
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
        MFe, MNi, MSi, MCO, lpc, alpha, muoff, trise, uvcor, aNi = pars
        # Total white dwarf mass
        MWD = MFe + MNi + MSi + MCO
        # Default blob
        blob = dict(self._default_blob)
        blob.update({ 'MFe': MFe, 'MNi': MNi, 'MSi': MSi, 'MCO': MCO,
            'MWD': MWD, 'lpc': lpc, 'alpha': alpha, 'muoff': muoff,
            'aNi': aNi, 'trise': trise, 'uvcor': uvcor, 'fail': 'default' })
        # Energetics -- all energies in ergs
        try:
            Egrv = SNPhysics.yl05_bind(MWD, lpc)
        except BadInputError:
            Egrv = -np.inf
        Enuc = SNPhysics.mi09_Enuc(MFe, MNi, MSi, MCO)
        Ekin = Enuc - Egrv
        Enuc51, Egrv51, Ekin51 = Enuc/1e+51, Egrv/1e+51, Ekin/1e+51
        blob.update({ 'Egrv51': Egrv51, 'Enuc51': Enuc51, 'Ekin51': Ekin51 })
        # Get rid of unphysical fits with extreme prejudice
        if Egrv < 0:
            blob['fail'] = "negEgrv"
            return blob
        elif Ekin < 0:
            blob['fail'] = "negEkin"
            return blob
        # scaling velocity of the ejecta
        vsc = SNPhysics.Density.v_sc(MWD*2e+33, Ekin)
        # generalized q lookup, (aNi, MFe, MNi) => (aNi, mNilo, mNihi)
        # assuming no central 56Ni hole
        if self.holefe:
            # Approximately correct answer; this is horrible code but I'm in
            # the mood for "good enough" rather than "good" right now.
            # I've verified that this approximation is good enough.
            raNi = 1.5*aNi
            if MFe/MWD > raNi:
                mNilo = MFe/MWD
            else:
                mNilo = np.sqrt(4*raNi*MFe/MWD) - raNi
            mNihi = (MFe + MNi)/MWD
        else:
            mNilo, mNihi = -0.5, (MFe + MNi)/MWD
        q = self.qgp.eval([mNilo, mNihi, 0.7*aNi])
        t0 = Jeffery99.t0(q, MWD*2e+33, Ekin)
        # Update blob with physical solution
        blob.update({ 'vsc': vsc/1e+5, 't0': t0, 'Q': q, 'fail': None })
        return blob

    def logl(self, pars, blob=None):
        """Log likelihood *only* for PTSampler

        This assumes that all the parameters lie within bounds laid out in
        self._features.  Implicit bounds caused by physics *necessary* for the
        likelihood to make sense, e.g., binding energy, must be included here.
        """

        # Fill the blob first, if necessary
        if blob is None:
            blob = Scalzo12_MCMC.fillblob(self, pars)
        if blob['fail'] is not None:
            return -np.inf
        MNi, alpha, trise, t0, muoff, uvcor = [blob[f] for f in
            ('MNi', 'alpha', 'trise', 't0', 'muoff', 'uvcor')]
        # Model:  energy deposited, including alpha factor near max
        model = MNi * Arnett82.epsilon(self.tfit + trise, t0)
        model[self.tfit < 10.0] *= alpha
        # Data:  include distance modulus offset, plus UV and alpha near max
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
        MFe, MNi, MSi, MCO, lpc, alpha, muoff, trise, uvcor, aNi = pars
        # Total white dwarf mass
        MWD = MFe + MNi + MSi + MCO
        # Neutronization prior (Krueger+ 2012, Seitenzahl+ 2013)
        eta = MNi/(MFe + MNi)
        self.eta_Pmu, self.eta_Psig = SNPhysics.krueger12_eta(lpc)
        # Other prior terms P(theta) for ancillary parameters
        chpri  = ((alpha - self.alpha_Pmu) / self.alpha_Psig)**2
        chpri += ((trise - self.trise_Pmu) / self.trise_Psig)**2
        chpri += ((muoff - self.muoff_Pmu) / self.muoff_Psig)**2
        chpri += ((eta - self.eta_Pmu) / self.eta_Psig)**2
        chpri += ((aNi - self.aNi_Pmu) / self.aNi_Psig)**2
        chpri += ((MCO/MWD - self.fCO_Pmu) / self.fCO_Psig)**2
        # Extra prior:  M = 1.40 +/- 0.02 Mch
        # chpri += ((MWD - 1.40)/0.02)**2
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
        self.best_model_L[self.best_model_t < 10.0] *= \
                best_blob['alpha'] / (1.0 + best_blob['uvcor'])
        self.best_model_L *= 10**(0.4*best_blob['muoff'])

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
        fig.set_size_inches(9.0, 9.0)
        pypl.subplots_adjust(left=0.1, right=0.9, bottom=0.10, top=0.95,
                             wspace=0.35, hspace=0.35)
        if plotfname:
            pypl.savefig(plotfname, dpi=100)
        if showplots:
            pypl.show()
