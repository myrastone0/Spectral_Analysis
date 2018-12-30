A hodge podge of spectral analysis tools.
-----------------------------------------

These tools are specific to far-infrared data obtained with the Herschel Space
Telescope.

Data Retrieval
--------------
Jython scripts to retrieve data from the Herschel Science Archive (HSA).
Observations were made with the Photoconductor Array Camera and 
Spectrometer (PACS) onboard the Herschel Space Telescope. 
Scripts are to be run inside the Herschel Interactive Processing Environment (HIPE).
They can retrieve the rebinned Herschel Science Center (HSC) pipeline
processed data products of native resolution (9.4 x 9.4 arcsecond spaxels),
or they can interpolate to smaller spaxel sizes. Data cubes are saved to FITS files.



Spectrum Fitting
----------------
- continuum/baseline subtraction via polynomial or spline fitting
- profile fitting of atomic fine-structure emission lines and molecular
lambda doublet lines



Modeling
--------
- galaxy rotation curve
- photodissociation regions


Visualizations
--------------
- create RGB images from various wavelength observations
<embed src="https://raw.githubusercontent.com/myrastone0/Spectral_Analysis/atomicLines/m82/contPlots/1342186798_m82_oiii88_contSub.pdf" width="500" height="375" 
 type='application/pdf'>
![alt text](https://raw.githubusercontent.com/myrastone0/Spectral_Analysis/atomicLines/m82/contPlots/1342186798_m82_oiii88_contSub.pdf)