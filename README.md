# pyBoloSN

This code is a toolkit for building fast-running, flexible models of supernova light curves.  It implements semi-analytic prescriptions from the literature and embeds them within the `emcee` Monte Carlo sampler ([Foreman-Mackey et al. 2012](http://arxiv.org/abs/1202.3665)), representing ensemble-average physics (as informed, for example, by numerical simulations of supernovae) using explicit or implicit prior constraints on parameters.  The goal is to promote feedback between supernova theory and observation by testing whole classes of models quickly on large datasets.

Physics prescriptions available in pyBoloSN include:
* Parametrized photospheric-phase bolometric light curves ([Arnett 1982](http://adsabs.harvard.edu/abs/1982ApJ...253..785A))
* Gray Compton scattering in late-time bolometric light curves ([Jeffery 1999](http://adsabs.harvard.edu/abs/1999astro.ph..7015J))
* Neutronization vs. white dwarf central density in type Ia supernovae ([Seitenzahl et al. 2011](http://adsabs.harvard.edu/abs/2011MNRAS.414.2709S), [Krueger et al. 2012](http://adsabs.harvard.edu/abs/2012ApJ...757..175K))
* Binding energy for differentially rotating white dwarfs ([Yoon & Langer 2005](http://adsabs.harvard.edu/abs/2005A%26A...435..967Y))

The original pyBoloSN code base was used in [Scalzo et al. (2014b)](http://adsabs.harvard.edu/abs/2014MNRAS.440.1498S); please cite this paper if you find this code useful, pending publication of a more formal methods paper for pyBoloSN.

Quick start guide
-----------------

Dependencies, which I recommend you handle with `conda`, include:
* `numpy`/`scipy`/`matplotlib` (as you might expect)
* `emcee` (at least v2.2.1)
* `sklearn` (no later than 0.17), since at present we're using the deprecated `sklearn.gaussian_process.GaussianProcess` regression class -- this needs fixing!

If you want to use the `Jeffery99.Qint` class to tabulate form factors for <sup>56</sup>Ni distribution (see [Jeffery 1999](http://adsabs.harvard.edu/abs/1999astro.ph..7015J)), you'll also need the VEGAS numerical integrator from `pygsl`; don't try this unless you know what you're doing (you might want to email me about it).

For a quick out-of-the-box start, suppose we want to reproduce the results for SN 2005el using the light curves given in the Scalzo+ 2014b data release, hosted [here](https://snfactory.lbl.gov/snf/data/SNfactory_Scalzo_etal_2014_DR.tar.gz) by the SNfactory collaboration.  To do this we might type

    run_S12_mcmc.py path/to/SN2005el/bololc_ebmv=*.txt --trise=17.0 --trise_err=2.0 --alpha=1.2 --alpha_err=0.2 --mu_off=0.00 --mu_err=0.13 --qtabfname=Jeffery99/j99_qgp_exp.pkl --nsamples=300000 --sampler=PTSampler

Breaking this down:

The list of light curves on the command line should represent a grid of reddening values for the same SN.  Since bolometric light curves have no color information on their own, the code handles host galaxy reddening by interpolating light curves across _E(B-V)_ (and/or _R<sub>V</sub>_).  The Scalzo+ 2014b format includes 40 light curves built in 0.01 mag increments of E(B-V) from 0.00-0.39.  All of these files should be included on the command line at present.

The various priors are Gaussian (_t<sub>rise</sub>_, &alpha;, &mu;, _E(B-V)_) or uniform ("uv_cor"); so for example the rise time has a Gaussian prior with mean `trise` and standard deviation `trise_err`.  The "qtabfn" setting should point to one of the pickle files in the `Jeffery99` subdirectory, which contain a look-up table of the <sup>56</sup>Ni form factor _Q_ as a function of the internal composition of the SN.

The code depends on the `emcee` package ([Foreman-Mackey et al. 2013](http://adsabs.harvard.edu/abs/2013PASP..125..306F)), downloadable [here](http://dfm.io/emcee/current/).  You can specify either "EnsembleSampler" or "PTSampler" on the command line, but the code defaults to PTSampler, which is what was used in Scalzo+ 2014b and is recommended for this problem.  Experience shows that the posterior for our model is multi-modal, with one set of solutions describing low-mass white dwarf progenitors (exploding through double detonations) and another set of solutions near the Chandrasekhar limit, so a solution like parallel tempering is needed to ensure that the chain mixes well.

Using pyBoloSN as a toolkit
---------------------------

The code can also be used as a toolkit, enabling the user to mix and match different assumptions about the internal structure of the white dwarf progenitors (or, potentially in the future, other types of exploding stars).  Creating one's own model will involve subclassing the `SNMCMC` class and adding appropriate settings.  The `Scalzo12` subdirectory has some templates to follow in this case.  The other subdirectories contain other building blocks:  `Arnett82` contains an implementation of the Arnett (1982) light curve model; `Jeffery99` contains code to numerically tabulate the form factors for <sup>56</sup>Ni distribution described in Jeffery (1999); and other, simpler priors can be found in `SNPhysics`.

**WARNING:** Some of the SNMCMC template classes have somewhat misleading names, since many modules are named by author-year and the resulting papers took longer for me to publish than I had really hoped.  Since this has caused some confusion let me set the record straight here:

* **The results of [Scalzo+ (2012)](http://adsabs.harvard.edu/abs/2012ApJ...757...12S), about super-Chandra candidate SNe Ia with velocity plateaus, were obtained from an old C++ code that was never made public.**  It used a basic Metropolis sampler and didn't do parallel tempering, although this shouldn't cast too much doubt on the results since nearly all the probability lies on high-mass solutions for these SNe.

* **The results from [Scalzo+ (2014b)](http://adsabs.harvard.edu/abs/2014MNRAS.440.1498S) were obtained using `Scalzo12/Scalzo12.py`.**  This model selects specific points from the bolometric light curve to include in the fit, using only one at-maximum point and any points more than 40 days past bolometric maximum, and puts a prior over the rise time.  The rise-time prior and the at-max point constrain the <sup>56</sup>Ni mass, while the late-time light curve constrains the ejected mass (conditional on the <sup>56</sup>Ni content and its distribution).

* **The contents of `Scalzo12/Scalzo14.py` include an experimental class not yet used in any published papers.**  The class uses the Arnett (1982) semi-analytic light curve model as the likelihood, but incorporates assumptions about gamma-ray transparency based on the 56Ni distribution as set out in Jeffery (1999).  This is possible because Arnett simply parametrizes gamma-ray transparency in terms of a timescale similar to Jeffery's _t<sub>0</sup>_ parameter, while Jeffery goes on to derive an expression for that parameter in terms of the ejecta structure.  This model is thus perhaps closer in its aims to the more rigorous semi-analytic models of [Pinto & Eastman (2000a)](http://adsabs.harvard.edu/abs/2000ApJ...530..744P).  Light curve points between 10 and 40 days after bolometric maximum light are excluded, since the radiation transfer assumptions made by Arnett and/or Jeffery are sure to break down in this regime.

License
-------

Copyright 2015 Richard Scalzo and contributors.

pyBoloSN is free software made available under the MIT License. For details see the LICENSE file.
