import torch
from torch import Tensor
from typing import Union, Sequence
from mirtorch.alg import CG, FISTA, POGM, power_iter
from mirtorch.linear import LinearMap, FFTCn, NuSense, Sense, FFTCn, Identity, Diff2dgram, Gmri, Wavelet2D, \
    NuSenseGram, Diff3dGram


class NuSense_om(LinearMap):
    '''
    The operator $A$. Provides gradient w.r.t $x$ and $\omage$.
    Attributes:
        smaps: sensitivity maps in [nbatch, nc, nx, ny, (nz)]
        traj: sampling trajectory in size [dim, npoints]
        norm: 'ortho' or None provided by torchkbnufft
        numpoints: number of interpolation points in nufft
        grid_size: oversampling rate in nufft
    '''
    def __init__(self,
                 smaps: Tensor,
                 traj: Tensor,
                 norm='ortho',
                 numpoints: Union[int, Sequence[int]] = 6,
                 grid_size: float = 2):
        self.traj = traj
        self.A = NuSense(smaps, traj, norm=norm, batchmode=True, numpoints=numpoints, grid_size=grid_size,
                         sequential=False)
        im_size = smaps.shape[2:]

        class Aforward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, ktraj, A):
                ctx.save_for_backward(ktraj, x)
                ctx.A = A
                return A * x

            @staticmethod
            def backward(ctx, y):
                # use in-plane opeators
                ktraj, xforward = ctx.saved_tensors
                dldom = torch.zeros_like(ktraj)
                for i in range(len(im_size)):
                    if len(im_size) == 1:
                        r = torch.arange(im_size[i]).expand(1, 1, -1).to(device=y.device)
                    if len(im_size) == 2:
                        r = torch.arange(im_size[i]).expand(1, 1, 1, -1).permute(0, 1, 3 - i, 2 + i).to(device=y.device)
                    if len(im_size) == 3:
                        r = torch.arange(im_size[i]).expand(1, 1, 1, 1, -1).to(device=y.device)
                        if i == 0:
                            r = r.permute(0, 1, 4, 2, 3)
                        if i == 1:
                            r = r.permute(0, 1, 2, 4, 3)
                        if i == 2:
                            r = r.permute(0, 1, 2, 3, 4)
                    xrd = xforward * r
                    AFxr = ctx.A * xrd
                    del xrd
                    dldwd = (torch.conj(AFxr.mul_(0 - 1j)).mul_(y)).real
                    dldom[:, i, :] = torch.sum(torch.sum(dldwd, dim=1), dim=0)
                return ctx.A.H * y, dldom, None

        self.forward_op = Aforward.apply

        class Aadjoint(torch.autograd.Function):
            @staticmethod
            def forward(ctx, y, ktraj, A):
                ctx.save_for_backward(ktraj, y)
                ctx.A = A
                return A.H * y

            @staticmethod
            def backward(ctx, x):
                ktraj, yforward = ctx.saved_tensors
                dldom = torch.zeros_like(ktraj)
                for i in range(len(im_size)):
                    if len(im_size) == 1:
                        r = torch.arange(im_size[i]).expand(1, 1, -1).to(device=x.device)
                    if len(im_size) == 2:
                        r = torch.arange(im_size[i]).expand(1, 1, 1, -1).permute(0, 1, 3 - i, 2 + i).to(device=x.device)
                    if len(im_size) == 3:
                        r = torch.arange(im_size[i]).expand(1, 1, 1, 1, -1).to(device=x.device)
                        if i == 0:
                            r = r.permute(0, 1, 4, 2, 3)
                        if i == 1:
                            r = r.permute(0, 1, 2, 4, 3)
                        if i == 2:
                            r = r.permute(0, 1, 2, 3, 4)
                    diagrx = x * r
                    Adiagrx = ctx.A * diagrx
                    del diagrx
                    dldwd = (Adiagrx.mul_(torch.conj(yforward) * (0 - 1j))).real
                    dldom[:, i, :] = torch.sum(torch.sum(dldwd, dim=1), dim=0)
                return ctx.A * x, dldom, None

        self.adjoint_op = Aadjoint.apply
        super(NuSense_om, self).__init__(tuple(self.A.size_in), tuple(self.A.size_out), device=smaps.device)

    def _apply(self, x):
        return self.forward_op(x, self.traj, self.A)

    def _apply_adjoint(self, y):
        return self.adjoint_op(y, self.traj, self.A)


class NuSenseGram_om(LinearMap):
    '''
    The Grame operator $A'A$, using the Toeplitz embedding. Provides gradient w.r.t x and omage
    Attributes:
        smaps: sensitivity maps in [nbatch, nc, nx, ny, (nz)]
        traj: sampling trajectory in size [dim, npoints]
        norm: 'ortho' or None provided by torchkbnufft
        numpoints: number of interpolation points in nufft
        grid_size: oversampling rate in nufft
    '''
    def __init__(self,
                 smaps: Tensor,
                 traj: Tensor,
                 norm='ortho',
                 numpoints: Union[int, Sequence[int]] = 6,
                 grid_size: float = 2):
        self.traj = traj
        batchmode = True
        self.A = NuSense(smaps, traj, norm=norm, batchmode=True, numpoints=numpoints, grid_size=grid_size)
        # self.AHA = self.A.H*self.A
        self.AHA = NuSenseGram(smaps, traj, norm=norm, batchmode=batchmode, numpoints=numpoints, grid_size=grid_size)
        im_size = smaps.shape[2:]

        class Aforward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, x, ktraj, AHA, A):
                ctx.save_for_backward(ktraj, x)
                ctx.AHA = AHA
                ctx.A = A
                return AHA * x

            @staticmethod
            def backward(ctx, dx):
                ktraj, xforward = ctx.saved_tensors
                dldom = torch.zeros_like(ktraj)
                Ax = ctx.A * xforward
                Adx = ctx.A * dx
                for i in range(len(im_size)):
                    if len(im_size) == 1:
                        r = torch.arange(im_size[i]).expand(1, 1, -1).to(device=dx.device)
                    if len(im_size) == 2:
                        r = torch.arange(im_size[i]).expand(1, 1, 1, -1).permute(0, 1, 3 - i, 2 + i).to(
                            device=dx.device)
                    if len(im_size) == 3:
                        r = torch.arange(im_size[i]).expand(1, 1, 1, 1, -1).to(device=dx.device)
                        if i == 0:
                            r = r.permute(0, 1, 4, 2, 3)
                        if i == 1:
                            r = r.permute(0, 1, 2, 4, 3)
                        if i == 2:
                            r = r.permute(0, 1, 2, 3, 4)
                    xr = xforward * r
                    Axr = ctx.A * xr
                    Ardx = ctx.A * (r * dx)
                    del xr
                    dldwd = (torch.conj(Axr).mul_(Adx).mul_(0 + 1j).add_(Ardx.mul_(torch.conj(Ax)).mul_((0 - 1j)))).real
                    dldom[:, i, :] = torch.sum(torch.sum(dldwd, dim=1), dim=0)
                return ctx.AHA * dx, dldom, None, None

        self.forward_op = Aforward.apply

        super(NuSenseGram_om, self).__init__(tuple(self.A.size_in), tuple(self.A.size_in), device=smaps.device)

    def _apply(self, x):
        return self.forward_op(x, self.traj, self.AHA, self.A)

    def _apply_adjoint(self, y):
        return self.forward_op(y, self.traj, self.AHA, self.A)


class Gram_inv(LinearMap):
    '''
    The inverse operator $(A'A+\lambdaI(^{-1}$, using the conjugate gradient to resolve. Provides gradient w.r.t x and omage
    Attributes:
        smaps: sensitivity maps in [nbatch, nc, nx, ny, (nz)]
        traj: sampling trajectory in size [dim, npoints]
        norm: 'ortho' or None provided by torchkbnufft
        numpoints: number of interpolation points in nufft
        grid_size: oversampling rate in nufft
    '''
    def __init__(self,
                 smaps: Tensor,
                 traj: Tensor,
                 lambd: float,
                 tol: float,
                 max_iter=100,
                 norm='ortho',
                 numpoints: Union[int, Sequence[int]] = 6,
                 grid_size: float = 2,
                 alert=False,
                 x0=None
                 ):
        self.traj = traj
        batchmode = True
        self.A = NuSense(smaps, traj, norm=norm, batchmode=batchmode, numpoints=numpoints, grid_size=grid_size)
        im_size = smaps.shape[2:]
        I = Identity(self.A.size_in)
        # self.AHA = self.A.H*self.A + lambd * I
        self.AHA = NuSenseGram(smaps, traj, norm=norm, batchmode=batchmode, numpoints=numpoints, grid_size=grid_size) + lambd * I
        self.x0 = x0
        self.solver_forward = CG(self.AHA, max_iter=max_iter, tol=tol, alert=alert)

        class Aforward(torch.autograd.Function):
            @staticmethod
            def forward(ctx, b, ktraj, A, solver_forward, x0):
                # print('allocate before forward CG', torch.cuda.memory_allocated())
                Pb = solver_forward.run(x0, b)
                ctx.A = A
                ctx.save_for_backward(ktraj, b, Pb)
                ctx.solver_forward = solver_forward
                return Pb

            @staticmethod
            def backward(ctx, dx):
                ktraj, b, Pb = ctx.saved_tensors
                dldom = torch.zeros_like(ktraj)
                Pdx = ctx.solver_forward.run(torch.zeros_like(dx), dx)
                APb = ctx.A * Pb
                APdx = ctx.A * Pdx
                for i in range(len(im_size)):
                    if len(im_size) == 1:
                        r = torch.arange(im_size[i]).expand(1, 1, -1).to(device=dx.device)
                    if len(im_size) == 2:
                        r = torch.arange(im_size[i]).expand(1, 1, 1, -1).permute(0, 1, 3 - i, 2 + i).to(
                            device=dx.device)
                    if len(im_size) == 3:
                        r = torch.arange(im_size[i]).expand(1, 1, 1, 1, -1).to(device=dx.device)
                        if i == 0:
                            r = r.permute(0, 1, 4, 2, 3)
                        if i == 1:
                            r = r.permute(0, 1, 2, 4, 3)
                        if i == 2:
                            r = r.permute(0, 1, 2, 3, 4)
                    Pdxr = Pdx * r
                    APdxr = ctx.A * Pdxr
                    Pbr = Pb * r
                    APbr = ctx.A * Pbr
                    dldwd = (
                        torch.conj(APbr).mul_((0 - 1j)).mul_(APdx).add_(APdxr.mul_(torch.conj(APb)).mul_(0 + 1j))).real
                    dldom[:, i, :] = torch.sum(torch.sum(dldwd, dim=1), dim=0)

                return Pdx, dldom, None, None, None

        self.forward_op = Aforward.apply

        super(Gram_inv, self).__init__(tuple(self.A.size_in), tuple(self.A.size_in), device=smaps.device)

    def _apply(self, x):
        if self.x0 is None:
            x0 = torch.clone(x.detach())
        else:
            x0 = self.x0
        return self.forward_op(x, self.traj, self.A, self.solver_forward, x0)

    def _apply_adjoint(self, y):
        if self.x0 is None:
            x0 = torch.clone(y.detach())
        else:
            x0 = self.x0
        return self.forward_op(y, self.traj, self.A, self.solver_forward, x0)



