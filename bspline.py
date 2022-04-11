import torch
import torch.nn as nn
import time
import math
import scipy
import scipy.linalg

def bspline2_1ndsynth(coeff, nfe, dt, gam, ext):
    # Coeff: [Ncycle]: Contain coefficient. Default should be ones.
    # length: The length of time points
    num_kernels = coeff.shape[0]
    ti = torch.linspace(0, num_kernels, nfe + ext)
    ti = torch.max(ti, -1.5*torch.ones_like(ti))
    ti = torch.min(ti, (num_kernels + 0.5)*torch.ones_like(ti))
    ti = ti + 2
    n0 = torch.floor(ti - 0.5).to(dtype = torch.long)

    # Construct the quadratic b-spline kernel
    def b2f0(t):
        return 3/4 - torch.pow(t,2)
    def b2f1(t):
        return torch.pow(torch.abs(t) - 1.5, 2)/2
    B = torch.zeros(nfe + ext, num_kernels)
    for i in range(coeff.shape[0]):
        coeff_i = torch.zeros(coeff.shape[0]+5).to(dtype=coeff.dtype, device=coeff.device)
        coeff_i[i+2] = coeff[i]
        ft = coeff_i[0 + n0]*b2f1(ti-n0) + coeff_i[1 + n0]*b2f0(ti-(n0+1)) + coeff_i[2 + n0]*b2f1(ti-(n0+2))
        B[:,i] = ft

    dB1 = B[ext//2:nfe+ext//2-1, :] - B[ext//2+1:nfe+ext//2, :]
    dB2 = dB1[0:-1, :] - dB1[1:, :]
    (_, b1max) = torch.max(dB1, dim=0)
    (_, b1min) = torch.min(dB1, dim=0)
    (_, b2max) = torch.max(dB2, dim=0)
    (_, b2min) = torch.min(dB2, dim=0)
    b1ind = np.union1d(b1max.numpy(), b1min.numpy())
    b1ind = torch.tensor(b1ind)
    b2ind = np.union1d(b2max.numpy(), b2min.numpy())
    b2ind = torch.tensor(b2ind)
    dB1 = dB1[b1ind, :]/gam/dt
    dB2 = dB2[b2ind, :]/dt/gam/dt
    return B.permute(1,0), dB1.permute(1,0), dB2.permute(1,0)


class SamplingLayerBspline2D(nn.Module):
    """
        The unit system are as follow:
            k: cycle / cm
            g: Gauss / cm
            s: Gauss / cm / s
            dt: s
            gamma: Hz / Gauss
            fov/res: cm
        Input:
            init_traj: [ndim, nacq]
            num_shots: number of shots
            nechos: num of points per acquisition
            decim: decimation rate
        Return:
            traj: [ndim, nshot, nechos]
            grad: [nshot, num_kernels]
            slew: [nshot, num_kernels]
    """

    def __init__(self, num_shots, nfe, decim=4, gamma=4.2576e+03, dt=4e-6, res=0.1, init_traj=None, ndims=2, gpu_ids=[],
                 ext=40):
        super(SamplingLayerBspline2D, self).__init__()
        self.nfe = nfe
        self.num_kernels = nfe // decim
        self.ndims = ndims
        self.num_shots = num_shots
        self.coeff = torch.ones(self.ndims, num_shots, self.num_kernels).to(gpu_ids[0])
        self.kmax = 1 / res
        self.gamma = gamma
        self.dt = dt
        self.decim = decim

        if decim > 1:
            # zero-paddind length
            self.ext = ext
            # build B:
            self.B, self.dB1, self.dB2 = bspline2_1ndsynth(torch.ones(self.num_kernels), nfe, dt, gamma, self.ext)
            self.B = self.B.to(gpu_ids[0])
            self.dB1 = self.dB1.to(gpu_ids[0])
            self.dB2 = self.dB2.to(gpu_ids[0])
            B = self.B.permute(1, 0).cpu().numpy()

        # The shape of traj_ref should be [ndim, nshot, npoints]
        # Load the traj from pre-computed numpy file
        traj_ref = np.load(init_traj)
        if len(traj_ref.shape) == 2:
            traj_ref = np.reshape(traj_ref, (self.ndims, self.num_shots, self.nfe))
        traj_ref = traj_ref / np.pi * self.kmax / 2
        # No parameterization
        if decim == 1:
            self.coeff = torch.tensor(traj_ref).to(dtype=self.coeff.dtype, device=gpu_ids[0])
        else:
            for ii in range(self.ndims):
                for jj in range(self.num_shots):
                    traj_ref_i = np.zeros(nfe + self.ext)
                    traj_ref_i[0:self.ext // 2] = traj_ref[ii, jj, 0]
                    traj_ref_i[nfe + self.ext // 2:nfe + self.ext] = traj_ref[ii, jj, nfe - 1]
                    traj_ref_i[self.ext // 2:nfe + self.ext // 2] = traj_ref[ii, jj, :]
                    self.coeff[ii, jj, :] = torch.tensor(np.linalg.lstsq(B, traj_ref_i)[0]).to(
                        dtype=self.coeff.dtype, device=self.coeff.device)
        self.coeff = torch.nn.Parameter(self.coeff)

    def forward(self, _):
        # Extract the locations for maximum gradient and slew rate
        if self.decim == 1:
            self.traj = self.coeff * 1
            self.gradient = (self.traj[:, :, :-1] - self.traj[:, :, 1:]) / self.gamma / self.dt
            self.slew = (self.gradient[:, :, :-1] - self.gradient[:, :, 1:]) / self.dt
        else:
            self.traj = torch.matmul(self.coeff, self.B)[:, :, self.ext // 2:self.nfe + self.ext // 2]
            self.gradient = torch.matmul(self.coeff, self.dB1)
            self.slew = torch.matmul(self.coeff, self.dB2)


        self.traj = torch.reshape(self.traj, (self.ndims, self.num_shots * self.nfe)).unsqueeze(0)

        # transfer trajectory to om (normalization, between -pi to pi)
        self.traj = self.traj / self.kmax * 2 * np.pi
        return self.traj, self.gradient, self.slew

