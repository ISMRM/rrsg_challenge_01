Reproduciblity Challenge #1 (2019) - Arbitrary SENSE
====================================================

This is the code repository of the Magnetic Resonance Technology and Methods
Lab, Institute for Biomedical Engineering, ETH Zurich and University of Zurich

to reproduce the reconstruction results of

> [1] K. P. Pruessmann, M. Weiger, P. Börnert, and P. Boesiger (2001),
> Advances in sensitivity encoding with arbitrary k-space trajectories
> Magnetic Resonance in Medicine 46(4), 638-651

as put forwards by the ISMRM Reproducible Research Study Group.

The code is largely based on a tutorial developed by Bertram J. Wilm and Klaas P.
Pruessmann for an Educational Session on Reconstruction at ESRMRB 2012, in Lisbon, Portugal.


Contributors
------------

Several people contributed to this repository by code, feedback, review and sharing
insights of their cg-SENSE experience:

- Franz Patzig
- Lars Kasper
- Thomas Ulrich
- Maria Engel
- S. Johanna Vannesjo
- Markus Weiger
- David O. Brunner
- Bertram J. Wilm
- Klaas P. Pruessmann

External Code used (from Educational Session ESMRMB):
- Felix Breuer
- Brian Hargreaves


Getting Started
---------------

We provide two ways of evaluating our results, depending on whether you have
access to Matlab environment on your system:

### Matlab
1. Clone this repository into a project folder of your liking, e.g. via
   `git clone https://github.com/mrtm-zurich/rrsg-arbitrary-sense.git rrsg-arbitrary-sense`
   There are three datasets in the repository which all come with all interleaves (called spokes in radial scans)
   allowing for setting an arbitrary undersampling factor. The datsets are:
    - Brain: radial brain dataset as originally provided by the rrsg study group at ISMRM 2019
    - Heart: radial heart dataset as originally provided by the rrsg study group at ISMRM 2019
    - Spiral: spiral dataset (3 interleaves) provided by MRT group of ETH Zurich
2. Run `code/demoRecon.m`.
    - Script that shows a simple example of a demo recon. Data is loaded and reconstructed using the CGSense method.
    - Feel free to play around with some of the parameters like the undersampling factor and see how it changes the result.
3. Run `code/main.m`.
    - The main script calls three functions creating the Figures 4 to 6 from the original paper.
	- They should correspond to figure 4,5, and 6 in the original MRM paper.
	- The created Matlab Figures are saved as PNGs to the `results/` subfolder of the project folder.
4. Note: If you just want to try out the reconstruction pipeline (load data, get SENSE map, run recon)
   for one undersampling factor), run `demoRecon.m` instead of `main.m` in the
   `code` subfolder of the repository.

### Other
If you don't have a Matlab license, you can run this code online on
https://CodeOcean.com as a Compute Capsule with its own DOI:

[ISMRM 2019 RRSG Challenge: MRI Technology and Methods Lab, ETH Zurich, GitHub Submission](https://dx.doi.org/10.24433/CO.5840424.v1)

1. Explore the published results on the right tab from a verified run of the code.
2. Press Re-Run to see results unfold for yourself
    - you might have to sign up to CodeOcean to do so


Requirements
------------

This code is written in Matlab, R2018a, but most likely works with earlier or later versions.


Results
-------

In order to address the goals of the submission, we provide the following result files, reflecting the current state of the repository:

1. The algorithm including gridding and FFT, SENSE map calculation, deapodization,
   k-space filtering and ringing filter was implemented as in [1]. To try out the pipeline for a single dataset and
   reconstruction, run `demoRecon.m` in the main repository folder.
	- Note that the SENSE maps were calculated following
	> [2] D. O. Walsh, A. F. Gmitro, and M. W. Marcellin, Adaptive reconstruction of phased array MR imagery,
	> Magnetic Resonance in Medicine, vol. 43, no. 5, pp. 682–690, May 2000.
	- Only the central, fully sampled part of the provided radial data (about 60 samples of each spoke for the brain data) were utilized for this computation. For the heart dataset all samples were used to calculate the SENSE maps.
2. The figure reproducing reconstruction errors of R=2,3,4,5 undersampled brain
   data is provided in `results/Figure4_brain_logDeltaOverIterations.png`
    - Note that the upper-case delta, i.e., the relative deviation from the exact solution, cannot be computed in the absence of a ground truth for the imaged object. Therefore, we
	reconstructed an image with R=1 and picked it as the ground truth for calculating the reconstruction error for R>2.
3. Reconstruction results for R=1,2,3,4 undersampled brain data are stored in
   two PNG files for better readability,
   `results/Figure5_brain_undersamplingRecon_part{1,2}.png`
4. Reconstructions of the provided cardiac data using the first 11,22,33,44,55
   spokes (R=5,4,3,2,1 respectively) are provided in
   `results/Figure6_heart_undersamplingRecon.png`.


Figures
-------

### Figure 4
![Figure 4](results/Figure4_brain_logDeltaOverIterations.png?raw=true "Figure 4")

### Figure 5
![Figure 5, part 1](results/Figure5_brain_undersamplingRecon_part1.png?raw=true "Figure 5, part 1")
![Figure 5, part 2](results/Figure5_brain_undersamplingRecon_part2.png?raw=true "Figure 5, part 2")

### Figure 6
![Figure 6](results/Figure6_heart_undersamplingRecon.png?raw=true "Figure 6")

Data structure
--------------

The input data is in h5 format. Two files are expected to be found:
(1) A file 'data' containing
    - the trajectory,
    - the complex signal data and
    - the image reconstruction size (not mandatory)
    The signal dimensions are:
    | — Sample Points — | — Interleaves — | — Coils — |

    The trajectory dimensions are:
    | — Sample Points — | — Interleaves — | — K-dimensions (=2, i.e. x and y as made for 2D data) — |
(2) A file 'sense' containing
    - the sense maps
    - a sense map mask
    - a noise covariance matrix (not mandatory)
