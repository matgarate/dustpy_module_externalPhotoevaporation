# External Photoevaporation Module for Dustpy

Includes the mass loss driven by external photoevaporation into [DustPy](https://github.com/stammler/dustpy) (Stammler and Birnstiel, [2022](https://ui.adsabs.harvard.edu/abs/2022ApJ...935...35S/abstract)), following the implementation of Sellek et al.[(2020)](https://ui.adsabs.harvard.edu/abs/2020MNRAS.492.1279S/abstract) implementation, using the [FRIED](www.friedgrid.com) grid to calculate the mass loss rates (Haworth et al., [2018](https://ui.adsabs.harvard.edu/abs/2018MNRAS.481..452H/abstract)).


To setup the external photoevaporation module, add the following lines after initialization (the FRIED table is required to use this module).

`from setup_externalPhotoevaporation import setup_externalPhotoevaporation_FRIED`

`setup_externalPhotoevaporation_FRIED(sim, fried_filename = "./friedgrid.dat", star_mass = 1., UV_flux = 1000.)`


See the `run_externalPhotoevaporation.py` code for an example.

If you use this module, please cite Gárate et al. (in prep.)
