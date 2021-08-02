import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import torch

from PiesArt import StimArtifactDataset
from PiesArt.ArtifactEraser.model import GRU_Denoiser
from torch.utils.data import DataLoader
from mef_tools.io import MefReader


Dat = StimArtifactDataset(1, ['MultiCenteriEEG_physiology', 'MultiCenteriEEG_pathology'], ['RCS'])
DLoad = DataLoader(Dat, batch_size=128, num_workers=8, shuffle=True)
model = GRU_Denoiser()


self = model
self.cuda(7)
optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)





for k, (x_orig, x_art, y_art) in enumerate(DLoad):
    optimizer.zero_grad()

    x_orig = x_orig.float().detach().to(self.device)
    x_art = x_art.float().detach().to(self.device)
    y_art = y_art.float().detach().to(self.device)

    x_rec, att = self(x_art)
    #break

    x_orig_fft = torch.fft.fft(x_orig, dim=-1).abs() / x_rec.shape[-1]
    x_rec_fft = torch.fft.fft(x_rec, dim=-1).abs() / x_rec.shape[-1]

    loss_signal = F.mse_loss(x_rec, x_orig)
    #loss_att = F.mse_loss(att, y_art)
    loss_att = F.binary_cross_entropy(att, y_art)
    loss_fft = F.mse_loss(x_rec_fft, x_orig_fft)
    loss = loss_signal + loss_att + loss_fft
    loss.backward()
    optimizer.step()

    if k % 50 == 0:
        print(k, loss.item(), loss_signal.item(), loss_att.item(), loss_fft.item())

        plt.subplot(3,1,1)
        plt.plot(x_orig[0].squeeze().detach().cpu().numpy())
        plt.plot(x_rec[0].squeeze().detach().cpu().numpy())

        plt.legend(['Original', 'Rec'])
        plt.title(str(k))


        plt.subplot(3,1,2)
        plt.plot(x_art[0].squeeze().detach().cpu().numpy())

        plt.subplot(3,1,3)
        plt.plot(y_art[0].squeeze().detach().cpu().numpy())
        plt.plot(att[0].squeeze().detach().cpu().numpy())
        plt.show()




model.eval().cpu()
#Rdr = MefReader('/mnt/Helium/filip/Projects/2021_Sleep_ERP_GS/M5/data/M5_Day_0_2Hz/ieeg M5_Day_0_2Hz.mefd')
Rdr = MefReader('/mnt/Helium/filip/Projects/2021_Sleep_ERP_GS/M1/data/M1_Day_2_2Hz/ieeg M1_Day_2_2Hz.mefd')

s = Rdr.get_property('start_time')[0]
x_real = Rdr.get_data('e12-e15', s, s + 10*1e6 )

var = np.nanstd(x_real)
mu = np.nanmean(x_real)

x_real = (x_real - mu) / var
x_real[np.isnan(x_real)] = 0

x_rec, yy = self(torch.tensor(x_real).float().view(1,1,-1))

plt.plot(x_real)
plt.plot(x_rec.squeeze().detach().numpy())
plt.show()

