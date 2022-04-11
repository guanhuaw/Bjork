### B-spline Parameterized Joint Optimization of Reconstruction and K-space Trajectories (BJORK) for Accelerated 2D MRI

This repo provides a PyTorch-based reimplementation of 'B-spline Parameterized Joint Optimization of Reconstruction and K-space Trajectories (BJORK) for Accelerated 2D MRI' and 'Efficient approximation of Jacobian matrices involving a non-uniform fast Fourier transform (NUFFT)'.

Since researchers' usually have personalized training pipeline, we currently provide the key components, the B-spline parameterization of trajectory and the efficient calculation of NUFFT Jacobian here. You may plug it into your own training modules with minimal modifications. However, if the demands for a runnable, self-contained code are high (please open an issue if needed), we will also provide whole package. 

`sys_ops.py` contains the forward mode, adjoint model, frame operaor, and the inverse $(A'A+\lambda I)^{-1}$. The Jacobian is approximated by the efficient algorithm detailed in 'Efficient approximation of Jacobian matrices involving a non-uniform fast Fourier transform (NUFFT)'. The implementation use the MRI system matrix provided by [MIRTorch](https://github.com/guanhuaw/MIRTorch).

`bspline.py` provides the B-spline parameterization of the sampling trajectory. 

`demo.py` provides the Non-Cartesian adaption of the [MoDL](https://arxiv.org/abs/1712.02862) reconstruction, which can be inserted it to [pix2pix training pipeline](https://github.com/guanhuaw/DeepMAGiC), with a user-defined dataloader. 

### Requirements

The minimum requirements are numpy, scipy, PyTorch 1.10 and [MIRTorch](https://github.com/guanhuaw/MIRTorch).

### Acknowledgments

If the code is helpful to your research, please consider citing:

```bibtex
@article{wang:22:bjork,
  author={Wang, Guanhua and Luo, Tianrui and Nielsen, Jon-Fredrik and Noll, Douglas C. and Fessler, Jeffrey A.},
  journal={IEEE Transactions on Medical Imaging}, 
  title={B-spline Parameterized Joint Optimization of Reconstruction and K-space Trajectories (BJORK) for Accelerated 2D MRI}, 
  year={2022},
  pages={1-1},
  doi={10.1109/TMI.2022.3161875}}
```

```bibtex
@article{wang:22:eaj,
  title={Efficient approximation of Jacobian matrices involving a non-uniform fast Fourier transform (NUFFT)},
  author={Wang, Guanhua and Fessler, Jeffrey A},
  journal={arXiv preprint arXiv:2111.02912},
  year={2021}
}
```

This reimplementation uses the [torchkbnufft](https://github.com/mmuckley/torchkbnufft/tree/main) toolbox:

```bibtex
@conference{muckley:20:tah,
  author = {M. J. Muckley and R. Stern and T. Murrell and F. Knoll},
  title = {{TorchKbNufft}: A High-Level, Hardware-Agnostic Non-Uniform Fast {Fourier} Transform},
  booktitle = {ISMRM Workshop on Data Sampling \& Image Reconstruction},
  year = 2020,
  note = {Source code available at https://github.com/mmuckley/torchkbnufft}.
}
```

