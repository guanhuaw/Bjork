def forward(self):
    # ktral: [nbatch, ndim, num_points]
    # grad/slew: [nbatch, ndim, nshot, nfe]
    self.ktraj, self.grad, self.slew = self.netSampling(1)
    A = NuSense_om(self.smap, self.ktraj, grid_size=self.opt.grid_size, norm='ortho', numpoints=self.opt.numpoints)
    ATA = NuSenseFrame_om(self.smap, self.ktraj, grid_size=self.opt.grid_size, norm='ortho', numpoints=self.opt.numpoints)
    self.kunder = A * self.Ireal
    if self.isTrain:
        self.kunder = self.kunder + self.opt.noise_level * torch.randn_like(self.kunder) * torch.max(
            torch.abs(self.kunder))
    self.Iunder = A.H * self.kunder
    self.P0 = Frame_inv(self.smap, self.ktraj, 0.001, self.opt.MODLtol, max_iter=12, norm='ortho',
                       numpoints=6, grid_size=2, alert=False)
    self.P1 = Frame_inv(self.smap, self.ktraj, self.opt.MODLLambda, self.opt.MODLtol, max_iter=8, norm='ortho',
                       numpoints=6, grid_size=2, alert=False)
    # Initializing estimiation with qwls
    Iinit = self.P0*self.Iunder
    Iiter = Init
    for ii in range(self.opt.num_blocks):
        Iiter = self.netG_I(torch.view_as_real(Iiter.squeeze(1)).permute(0, 3, 1, 2))
        Iiter = self.P1*(self.Iunder + self.opt.MODLLambda*torch.view_as_complex(Iiter.permute(0,2,3,1).contiguous()).unsqueeze(1))