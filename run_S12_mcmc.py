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

import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as pypl

from Scalzo12 import Scalzo12_MCMC, Scalzo12_MCMC_EBMV

# ----------------------------------------------------------------------------
#                               Main routine
# ----------------------------------------------------------------------------

def main():
    """The main routine."""

    parser = argparse.ArgumentParser(
            description="Bolometric light curve mass reconstruction.")
    parser.add_argument('bolo_fn', nargs='+',
                        help="Bolometric light curve text files (plural?)")
    parser.add_argument('--trise', type=float, default=17.0,
                        help="Bolometric rise time in days")
    parser.add_argument('--trise_err', type=float, default=17.0,
                        help="Bolometric rise time uncertainty in days")
    parser.add_argument('--alpha', type=float, default=1.2,
                        help="Radiation trapping factor for Arnett's rule")
    parser.add_argument('--alpha_err', type=float, default=0.2,
                        help="Radiation trapping factor uncertainty")
    parser.add_argument('--ebmv', type=float, default=0.00,
                        help="Host galaxy reddening")
    parser.add_argument('--ebmv_err', type=float, default=0.01,
                        help="Host galaxy reddening uncertainty")
    parser.add_argument('--mu_off', type=float, default=0.000,
                        help="Distance modulus offset to apply")
    parser.add_argument('--mu_err', type=float, default=0.001,
                        help="Distance modulus uncertainty")
    parser.add_argument('--aNi', type=float, default=0.2,
                        help="56Ni mixing scale (in mass units)")
    parser.add_argument('--aNi_err', type=float, default=0.1,
                        help="56Ni mixing scale uncertainty")
    parser.add_argument('--holefe', action='store_true', default=False,
                        help="Turn on 56Ni hole due to stable Fe production?")
    parser.add_argument('--uvcor_lo', type=float, default=0.0,
                        help="Lower bound on UV flux correction")
    parser.add_argument('--uvcor_hi', type=float, default=0.1,
                        help="Upper bound on UV flux correction")
    parser.add_argument('--qtabfname', default='qint_exp.pkl',
                        help="Pickled tabulated values of q form factor")
    parser.add_argument('--nsamples', type=int, default=100000,
                        help="Number of MCMC samples to draw")
    parser.add_argument('--ensemblesamp', action='store_true', default=False,
                        help="Use emcee EnsembleSampler instead of PTSampler")
    args = parser.parse_args()

    sn_name = re.search("\/(SN[F12].*|MPA.*|LSQ.*|PTF.*)\/",
                        args.bolo_fn[0]).group(1)

    # Load the light curve data.  These are assumed to have been prepared
    # in a very standard way -- all the same light curve points, but with
    # different values of the reddening on a grid.
    multi_f, multi_df, multi_ebmv = [ ], [ ], [ ]
    for fn in args.bolo_fn:
        bolo_t, bolo_f, bolo_df = np.loadtxt(fn, unpack=True, usecols=[0,1,4])
        ebmv = np.float(re.search("ebmv=(\d\.\d+)", fn).group(1))
        multi_f.append(bolo_f)
        multi_df.append(bolo_df)
        multi_ebmv.append(ebmv)

    # If there's only one light curve, no point in marginalizing over the
    # reddening so use the simpler version.
    if len(args.bolo_fn) == 1:
        mcmc = Scalzo12_MCMC(
                bolo_t, multi_f[0], multi_df[0], muoff_err=args.mu_err,
                trise=args.trise, trise_err=args.trise_err,
                alpha=args.alpha, alpha_err=args.alpha_err,
                uvcor_lo=args.uvcor_lo, uvcor_hi=args.uvcor_hi,
                aNi=args.aNi, aNi_err=args.aNi_err, holefe=args.holefe,
                qtabpkl=args.qtabfname, sn_name=sn_name)
    else:
        mcmc = Scalzo12_MCMC_EBMV(
                bolo_t, multi_f, multi_df, multi_ebmv, muoff_err=args.mu_err,
                ebmv=args.ebmv, ebmv_err=args.ebmv_err,
                trise=args.trise, trise_err=args.trise_err,
                alpha=args.alpha, alpha_err=args.alpha_err,
                uvcor_lo=args.uvcor_lo, uvcor_hi=args.uvcor_hi,
                aNi=args.aNi, aNi_err=args.aNi_err, holefe=args.holefe,
                qtabpkl=args.qtabfname, sn_name=sn_name)
    # Include a switch to run EnsembleSampler
    if args.ensemblesamp:
        print "WARNING: Using emcee.EnsembleSampler for this run"
        print "Recommended for testing only.  Sampling will run faster,"
        print "but this problem is multi-modal so watch out!"
        mcmc.run_mcmc(sampler='EnsembleSampler')
    else:
        print "Using emcee.PTSampler for this run"
        mcmc.run_mcmc(sampler='PTSampler')
    mcmc.show_results()

# ----------------------------------------------------------------------------
#                This thing will eventually be callable again
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()
