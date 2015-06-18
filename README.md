# pyBoloSN

This code is a toolkit for building fast-running, flexible models of supernova light curves.  It implements semi-analytic prescriptions from the literature and embeds them within the `emcee` Monte Carlo sampler ([Foreman-Mackey, Hogg, Lang & Goodman 2012]
(http://arxiv.org/abs/1202.3665)), representing ensemble-average physics (as informed, for example, by numerical simulations of supernovae) using explicit or implicit prior constraints on parameters.  The goal is to promote feedback between supernova theory and observation by testing whole classes of models quickly on large datasets.

Physics prescriptions available in pyBoloSN include:
* Parametrized photospheric-phase bolometric light curves ([Arnett 1982] (http://adsabs.harvard.edu/abs/1982ApJ...253..785A))
* Gray Compton scattering in late-time bolometric light curves ([Jeffery 1999] (http://adsabs.harvard.edu/abs/1999astro.ph..7015J))
* Neutronization vs. white dwarf central density in type Ia supernovae ([Seitenzahl et al. 2011] (http://adsabs.harvard.edu/abs/2011MNRAS.414.2709S), [Krueger et al. 2012] (http://adsabs.harvard.edu/abs/2012ApJ...757..175K))
* Binding energy for differentially rotating white dwarfs ([Yoon & Langer 2005] (http://adsabs.harvard.edu/abs/2005A%26A...435..967Y))

The original pyBoloSN code base was used in [Scalzo et al. (2014b)](http://adsabs.harvard.edu/abs/2014MNRAS.440.1498S); please cite this paper if you find this code useful, pending publication of a more formal methods paper for pyBoloSN.

License
-------

Copyright 2015 Richard Scalzo and contributors.

pyBoloSN is free software made available under the MIT License. For details see the LICENSE file.
