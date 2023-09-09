<img src="https://github.com/alexysong/inkstone/blob/main/figs/logo.png" align="middle" alt="logo" width="250">

**Inkstone** simulates the electromagnetic properties of 3D and 2D multi-layered structures with in-plane periodicity, such as gratings, photonic-crystal slabs, metasurfaces, vertical-cavity or photonic-crystal surface-emitting lasers (VCSEL, PCSEL), (patterned) solar cells, nano-antennas, and more.

Internally, Inkstone implements rigorous coupled-wave analysis (RCWA), a. k. a. Fourier Modal Method (FMM). 

### Inkstone can calculate: 
* the reflection, transmission, and absorption of the structure
* the total and by-order power fluxes of the propagating and the evanescent waves in each layer
* electric and magnetic field amplitudes at any locations in the structure,
* band-structures based on the determinant of the scattering matrix of the structure.

### Features of Inkstone:
* It supports efficient and flexible parameter-scanning. You can change part of your structure such as the shapes and sizes of some patterns, or some material parameters. Inkstone only recalculates the modified parts and produces the final results efficiently.
* It allows both tensorial permittivities and tensorial permeabilities, such as in anisotropic, magneto-optical, or gyromagnetic materials. 
* It can calculate the determinant of the scattering matrix on the complex frequency plane. 
* Pre-defined shapes of patterns can be used, including rectangular, parallelogram, disk, ellipse, 1D, and polygons. Closed-form Fourier transforms and corrections for Gibbs phenomena are implemented. 
* It is fully 3D.
* It is written in pure python, with heavy-lifting done in numpy and scipy.


## Quick Start
### Installation:

    $ pip install inkstone
Or,

    $ git clone git://github.com/alexysong/inkstone
    $ pip install .

### Usage

The [examples](examples/) folder contains various self-explaining examples to get you started.

## Dependencies

*   python 3.6+
*   numpy
*   scipy

## Units, conventions, and definitions

### Unit system
We adopt a natural unit system, where vacuum permittivity, permeability, and light speed are $\varepsilon_0=\mu_0=c_0=1$.

### Sign convention
Sign conventions in electromagnetic waves:

$$e^{i(kx-\omega t)}$$

where $k$ is the wavevector, $x$ is spatial location, $\omega$ is frequency, $t$ is time.

By this convention, a permittivity of $\varepsilon_r + i\varepsilon_i$ with $\varepsilon_i>0$ means material loss, and $\varepsilon_i<0$ means material gain.

### Coordinates, incident angles, and polarizations

<img src="https://github.com/alexysong/inkstone/blob/main/figs/PhC_slab_vector_incident.svg" alt="drawing" width="300">

(Inkstone, **In**cident $\bm{k}$ on **st**acked peri**o**dic **n**ano **e**lectromagnetic structures.)

$\theta$ is defined as the angle between the incident $\vec{k}$ and the normal to the $xy$ plane. $\phi$ is defined as the angle between the projection of $\vec{k}$ in plane and $\hat{x}$. 

$s$ polarization is when $\vec{E}$ field of the incoming plane wave is in the $xy$ plane. $p$ polarization is orthogonal to it. 

For $\theta=0$, $\phi=0$, $s$ is when $\vec{E}$ of incoming wave is along $\hat{y}$, $p$ is the orthogonal one. 

In 2d simulations (1d grating), the space is assumed to be in $x$ and $z$. Here, $s$ is when $\vec{E}$ of the incoming wave is in and out of the solving 2d space. $p$ is again orthogonal to it.

## What's new

*   Ver 0.3:

    Solved the convergence issue at Wood's anomaly. Now the calculation maintain the same convergence, stability, and speeed near and at Wood's anomaly.

## Citing
If you find Inkstone useful for your research, we would apprecite you citing our [paper](https://doi.org/10.1103/PhysRevLett.120.193903). For your convenience, you can use the following BibTex entry:

    @article{song2018broadband,
      title={Broadband Control of Topological Nodes in Electromagnetic Fields},
      author={Song, Alex Y and Catrysse, Peter B and Fan, Shanhui},
      journal={Physical review letters},
      volume={120},
      number={19},
      pages={193903},
      year={2018},
      publisher={American Physical Society}
    }

   
