#!/usr/bin/env python 

"""
RS 2013/07/09:  Monte Carlo Markov Chain base class for SN modeling
"""

# ----------------------------------------------------------------------------
#                               Dependencies
# ----------------------------------------------------------------------------

import acor
import emcee
import numpy as np
import matplotlib.pyplot as pypl
import cPickle as pickle
from scipy.stats import scoreatpercentile
from scipy.special import gammaincc, gammaln

from ..Utils import VerboseMsg, BadInputError

# ----------------------------------------------------------------------------
#                             Class definitions
# ----------------------------------------------------------------------------

def prob_contour_plot(x, y, xmin, xmax, nxbins, ymin, ymax, nybins):
    """Draws a contour plot of a probability density"""
    # Bin up the histogram
    data = np.array([x, y], dtype=np.float).T
    Z, edges = np.histogramdd(data, bins=(nxbins, nybins),
                              range=((xmin, xmax), (ymin, ymax)))
    Z = Z.T/Z.sum()
    # Levels for contours as in Scalzo+ 2010, 2012
    plevels = np.array([0.003, 0.01, 0.05, 0.10, 0.32, 1.0])
    Zlevels = 0.0*plevels
    colors = ['blue', 'cyan', 'lightgreen', 'yellow', 'red']
    # Calculate the proper contour levels so that they enclose the stated
    # integrated probability.
    Zcl = 0.0 
    for zi in sorted(Z.ravel()):
        Zcl += zi
        Zlevels[plevels > Zcl] = zi
        if Zcl > plevels[-2]:  break
    # Plot the filled contours and appropriate contour lines.
    Xm = 0.5*(edges[0][1:] + edges[0][:-1])
    Ym = 0.5*(edges[1][1:] + edges[1][:-1])
    X, Y = np.meshgrid(Xm, Ym)
    Zlevels[-1] = 1.0
    pypl.contourf(X, Y, Z, Zlevels, colors=colors)
    pypl.contour(X, Y, Z, Zlevels, linewidths=2, linestyles='-', colors='k')

def gammainccln(a, x, tol=1e-50):
    """Asymptotic series expansion of ln(gammaincc(a, x)) for large x"""
    return (a-1)*np.log(x) - x + np.log(1 + (a-1)/x) - gammaln(a)
    dfac, dfacm1, fsum, N = 1.0, 1.0, 0.0, max(a - 1, 0)
    # max 50 terms, probably not worth more
    for n in range(0, 50):
        fsum += dfac
        dfac *= (a - 1 - n)/x
        if abs(dfac) < tol or abs(dfac) > abs(dfacm1):  break
        dfacm1 = dfac
    return (a - 1)*np.log(x) - x + np.log(fsum) - gammaln(a)


class SNMCMC(object):
    """Class to encapsulate a basic MCMC parameter search

    I'm writing this as a plug-and-play interface to emcee for coming up with
    supernova models.  Realistically, emcee is pretty easy to use already,
    but there are certain things like setting up the initial state, bounds on
    parameters, etc., which can be streamlined by doing them in this way.
    """

    # Features of standard blob of chain, with upper and lower bounds
    _features = \
    [
       # name,     def,  blo,  hlo,  hhi,  bhi,  res,  fmt, label, e.g.:
       # 'MWD':    ( 1.4,  0.8,  0.8,  2.8,  2.8, 0.05, "{0:8.2f}",
       #            "Ejected mass (M$_\\odot$)"),
    ]
    # Names of features to be used as main MCMC parameters
    _mcmcpars = [ ]
    # Names of features for which to report marginalized confidence intervals
    _confints = [ ]
    # Description of subplots to plot when MCMC is done running
    _subplot_layout = (1, 1)
    _subplots = [ ]
    _contlvls = [0.01, 0.05, 0.16, 0.50, 0.84, 0.95, 0.99]
    # Default keywords for __init__, with default values
    _init_kwdef = { }

    def __init__(self, verbose=True, **init_kwargs):
        """Initialize the object"""
        # Unpack kwargs
        kw = dict(self._init_kwdef)
        kw.update(init_kwargs)
        # Features to show in histogram summary
        clname = self.__class__.__name__
        featureset = set(self._features.keys())
        # Check all the damn inputs to make sure they make sense.
        mcmcpset = set(self._mcmcpars)
        if not mcmcpset <= featureset:
            raise BadInputError("user supplied MCMC parameters {0}"
                                " that are not in {1} feature {2}".format(
                                mcmcpset - featureset, clname, featureset))
        subplfset = [ ]
        for subtuple in self._subplots:  subplfset += subtuple
        subplfset = set(subplfset)
        if not subplfset <= featureset:
            raise BadInputError("user supplied plot features {0}"
                                " that are not in {1} feature {2}".format(
                                subplfset - featureset, clname, featureset))

        conflset = set(self._confints)
        if not conflset <= featureset:
            raise BadInputError("user supplied confidence region features {0}"
                                " that are not in {1} feature {2}".format(
                                conflset - featureset, clname, featureset))
        nsubplay, nsubplots = np.prod(self._subplot_layout), len(self._subplots)
        if nsubplay < nsubplots:
            raise BadInputError("user supplied plot layout {0} w/{1} subplots"
                                " but only {2} subplots specified".format(
                                self._subplot_layout, nsubplay, nsubplots))
        for f, fv in self._features.items():
            if not np.all([# fv[0] >= fv[1], fv[0] <= fv[4],
                           fv[1] <= fv[4], fv[2] <= fv[3],
                           fv[5] <= 0.1*(fv[3]-fv[2])]):
                print "uh oh:  ", fv
                raise BadInputError("bad value boundaries in {0}._features"
                                    " for feature {1}".format(clname, f))

        # ok!  finally time to move
        self.p0 = np.array([self._features[f][0] for f in self._mcmcpars])
        self.plo = np.array([self._features[f][1] for f in self._mcmcpars])
        self.phi = np.array([self._features[f][4] for f in self._mcmcpars])
        self.psig = np.array([self._features[f][5] for f in self._mcmcpars])/2
        # make sure we're not too close to the boundaries
        blo, bhi = self.plo + 3*self.psig, self.phi - 3*self.psig
        self.p0[self.p0 < blo] = blo[self.p0 < blo]
        self.p0[self.p0 > bhi] = bhi[self.p0 > bhi]
        # mop-up
        if not hasattr(self, 'vmsg'):
            self.vmsg = VerboseMsg(prefix=clname, verbose=verbose)
        self._default_blob = dict(
                [(f, v[0]) for f, v in self._features.items()])

    def lnPchisq(self, chisq, ndof):
        """Shortcut to log(gamma function) for chi-square fit probability
        
        If the chi-square is less than 1000, evaluates the probability of
        a good fit for N degrees of freedom using scipy.special.gammaincc().
        Otherwise, uses an asymptotic series to prevent underflow.
        """
        if chisq < 1000.0:
            return np.log(gammaincc(0.5*ndof, 0.5*chisq))
        else:
            return gammainccln(0.5*ndof, 0.5*chisq)

    def isgood(self, _lnprob, blob):
        """Should this blob should be counted in the final results?
        
        SNMCMC is meant to be a virtual class, so the user needs to define
        this for their problem in derived classes.
        """
        return _lnprob > -50

    def partest(self, pars):
        """Runs simple parameter test to see if we get the answer we expect"""
        lnp, blob = self.lnprob(pars)
        print "partest:  pars =", pars
        print "partest:  blob =", blob
        print "partest:  logp =", lnp

    def logl(self, pars, blob=None):
        """Log likelihood *only* for PTSampler

        This assumes that all the parameters lie within bounds laid out in
        self._features.  Implicit bounds caused by physics *necessary* for the
        likelihood to make sense, e.g., binding energy, must be included here.
        SNMCMC is meant to be a virtual class, so the user needs to define
        this for their problem in derived classes.
        May also use and modify a blob if one is passed in.
        """
        return 0.0

    def logp(self, pars, blob=None):
        """Log prior *only* for PTSampler

        This assumes that all the parameters lie within bounds laid out in
        self._features.  Implicit bounds caused by physics assumed beyond
        what's needed to calculate the likelihood, e.g., neutronization,
        must be included here.  SNMCMC is meant to be a virtual class, so
        the user needs to define this for their problem in derived classes.
        May also use and modify a blob if one is passed in.
        """
        return 0.0

    def _bounded_logp(self, pars, blob=None):
        """Evaluate the log prior within given bounds
        
        This wraps the virtual method logp(), enforcing boundaries on
        parameters.  Parameter bounds could be considered part of the prior.
        """
        if not np.all([p > plo for p, plo in zip(pars, self.plo)] +
                      [p < phi for p, phi in zip(pars, self.phi)]):
            return -np.inf
        else:
            return self.logp(pars)

    def lnprob(self, pars):
        """Log likelihood + prior for EnsembleSampler

        Splitting up lnprob = logl + logp makes more sense for PTSampler,
        particularly because it doesn't support blobs.  For EnsembleSampler
        we put them back together.
        """
        # Bayes' theorem:  P(theta|y) = P(y|theta)*P(theta)/P(y)
        # Set P(y) = 1 since those are the data we in fact observed
        # Thus ln P(theta|y) = ln P(y|theta) + ln P(theta)
        blob = self.fillblob(pars)
        _logp = self.logp(pars, blob)
        _logl = self.logl(pars, blob)
        return _logl + _logp, blob

    def _bounded_lnprob(self, pars):
        """Log likelihood + prior for EnsembleSampler, sampled within bounds"""
        blob = self.fillblob(pars)
        _logp = self._bounded_logp(pars, blob)
        if _logp == -np.inf:
            _logl = -np.inf
        else:
            _logl = self.logl(pars, blob)
        return _logl + _logp, blob

    def _acorthin(self):
        """Use autocorrelation time of chain(s) to guess optimal thinning"""
        try:
            if isinstance(self.sampler, emcee.PTSampler):
                # We're most interested in the zero-temperature chain here
                _acor = self.sampler.acor[0]
            else:
                # Just use everything
                _acor = self.sampler.acor
            return min(int(np.round(np.median(_acor))), 10)
        except Exception as e:
            self.vmsg("{0} while calculating acor: {1}".format(
                e.__class__.__name__, e))
            return 3

    def _guess(self, nwalkers):
        """Makes an initial guess around the default position"""
        return emcee.utils.sample_ball(self.p0, self.psig, nwalkers)

    def run_mcmc(self, nsamples=100000, sampler="EnsembleSampler"):
        """Uses emcee.EnsembleSampler to sample our model's posterior
        
        Makes some sensible guesses about how to accomplish what we want,
        namely, to achieve some guaranteed specified number nsamples of
        independent Monte Carlo samples of our problem's posterior.
        -- Total number of samples is nwalkers x ndim x niter >= nsamples
        -- To be within 10% of goal, check every 10% of iterations = 1 round
        -- Use nwalkers = 20 x ndim and burn in for 1 round
        -- If acceptance < 0.05 after 5 non-burn-in rounds, abort with error
        """

        ndim = len(self.p0)
        nwalkers = 20*ndim
        if nsamples < 100*nwalkers:  nsamples = 100*nwalkers
        niter = nsamples / (10*nwalkers)
        if sampler == "EnsembleSampler":
            pars = self._guess(nwalkers)
            self.sampler = emcee.EnsembleSampler(
                     nwalkers, ndim, self._bounded_lnprob)
        elif sampler == "PTSampler":
            ntemps = 20
            betas = np.array([1.0/2**(0.5*n) for n in range(ntemps)])
            pars = np.array([self._guess(nwalkers) for T in range(ntemps)])
            self.sampler = emcee.PTSampler(ntemps, nwalkers, ndim,
                    self.logl, self._bounded_logp, betas=betas)
        else:
            raise BadInputError("'sampler' parameter to run_mcmc must be in "
                                "['EnsembleSampler', 'PTSampler']")

        # Burn in the chain.  The burning-in process means a lot of the early
        # samples will be strongly correlated, but a well-burned-in chain
        # should be uncorrelated every 2-3 steps, so start with thin = 3.
        # Really burn the damn thing in!
        pars = self.sampler.run_mcmc(pars, 10*niter, thin=10)[0]
        thin = thin0 = 3
        self.vmsg("Starting run with thin =", thin)
        stopme = False
        self.sampler.reset()
        while True:
            try:
                self.sampler.run_mcmc(pars, niter*thin, thin=thin)
            except MemoryError:
                # If we run out of memory, just use what we've got
                stopme = True
            if sampler == "EnsembleSampler":
                # Just retrieve the samples
                nblobs = np.prod(self.sampler.chain[0].shape)/ndim
                self.lnproblist = np.array(self.sampler.lnprobability).ravel()
                self.bloblist = np.array(self.sampler.blobs).ravel()
                faccept = np.median(self.sampler.acceptance_fraction)
            elif sampler == "PTSampler":
                # Reincarnate zero-temp blobs, which PTSampler doesn't support
                nblobs = np.prod(self.sampler.chain[0].shape)/ndim
                bpars = self.sampler.chain[0].reshape(nblobs, ndim)
                self.lnproblist = np.array(self.sampler.lnprobability[0]).ravel()
                self.bloblist = np.array([self.fillblob(p) for p in bpars])
                faccept = np.median(self.sampler.acceptance_fraction[0])
            else:
                pass

            self.goodidx = np.array([self.isgood(lnp, b) for lnp, b in
                                     zip(self.lnproblist, self.bloblist)])
            nblobs = len(self.goodidx)
            ngood = sum(self.goodidx)
            if ngood > nsamples or stopme:
                self.vmsg("Quitting with {0} good blobs".format(ngood))
                break
            elif len(self.bloblist) > 0.5*nsamples and faccept < 0.05:
                self.vmsg("acceptance fraction = {0}, convergence sucks"
                          .format(faccept))
                self.vmsg("bailing after {0} samples, don't trust results"
                          .format(nblobs))
                break
            else:
                self.vmsg("Chain has {0} good blobs so far".format(ngood))
                self.vmsg("lnprob min, max = {0}, {1}".format(
                    np.min(self.lnproblist), np.max(self.lnproblist)))
                self.vmsg("acceptance fraction =", faccept)
                """
                failtot = { }
                for b in self.bloblist:
                    if b['fail'] not in failtot:  failtot[b['fail']] = 0
                    failtot[b['fail']] += 1
                self.vmsg("Fail totals:", failtot)
                """
                thin = self._acorthin()
                if thin != thin0:
                    self.vmsg("Adjusting thin ~ tau =", thin)
                    thin0 = thin

    def show_results(self, makeplots=True, showplots=True, plotfname=None):
        """Display results of Markov chain sampling.
        
        Shows marginalized confidence intervals on key parameters, as well as
        full histograms and contour plots of 2-D joint confidence regions.
        """

        # If we're going to save or show plots, we have to show them first
        if plotfname or showplots:  makeplots = True

        # Unpack the good blobs into plot-ready numpy arrays
        if sum(self.goodidx) < 5:
            self.vmsg("No good blobs, hence no results to show!")
            return
        goodprobs = self.lnproblist[self.goodidx]
        goodblobs = self.bloblist[self.goodidx]
        allsamp = dict([(feature, np.array([b[feature] for b in goodblobs]))
                        for feature in goodblobs[0]])
        print self.__class__.__name__, "fit results:"
        print "   len(bloblist)  =", len(self.bloblist)
        print "   len(goodblobs) =", len(goodblobs)
        print "   physical frac. =", len(goodblobs)*1.0/len(self.bloblist)
        print "   accepted frac. =", np.median(self.sampler.acceptance_fraction)
        self.lnprobs, self.blobs = goodprobs, goodblobs
        for lnp, b in zip(self.lnprobs, self.blobs):
            b['lnprob'] = lnp
        # max probability
        Pfit_max = np.exp(np.max(goodprobs))
        print "   best fit prob. =", Pfit_max
        # feature quantiles
        self.pctiles = { }
        for f in self._confints:
            self.pctiles[f] = np.array(
                    [scoreatpercentile(allsamp[f], 100.0*q)
                     for q in self._contlvls], dtype=np.float).ravel()
        print "Attr.   ", " ".join(["{0:8.2f}".format(p) for p in self._contlvls])
        for f in self._confints:
            print "{0:8s}".format(f),
            print " ".join([self._features[f][6].format(p)
                            for p in self.pctiles[f]])

        # make plots
        if not makeplots:  return
        pypl.figure()
        for i in range(len(self._subplots)):
            pypl.subplot(self._subplot_layout[0], self._subplot_layout[1], i+1)
            if len(self._subplots[i]) == 1:
                f = self._subplots[i][0]
                c = self._features[f]
                data = allsamp[f]
                hlo, hhi, res, fmt, label = c[2], c[3], c[5], c[6], c[7]
                nbins = int(round((hhi-hlo)/(1.0*res)))
                pypl.hist(data, nbins, range=(hlo, hhi), normed=1,
                          facecolor='green', alpha=0.75)
                pypl.xlabel(label)
            elif len(self._subplots[i]) == 2:
                fx, fy = self._subplots[i]
                cx, cy = self._features[fx], self._features[fy]
                dx, dy = allsamp[fx], allsamp[fy]
                hxlo, hxhi, xres, xfmt, xlabel = [cx[j] for j in (2,3,5,6,7)]
                hylo, hyhi, yres, yfmt, ylabel = [cy[j] for j in (2,3,5,6,7)]
                nxbins = int(round((hxhi-hxlo)/(1.0*xres)))
                nybins = int(round((hyhi-hylo)/(1.0*yres)))
                prob_contour_plot(dx, dy, hxlo, hxhi, nxbins, hylo, hyhi, nybins)
                pypl.xlabel(xlabel)
                pypl.ylabel(ylabel)
            else:
                raise BadInputError("bad entry #{0} in {1}._subplots".format
                                    (i, self.__class__.__name__))
        if showplots:
            pypl.show()
        if plotfname:
            pypl.savefig(plotfname)
