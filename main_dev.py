import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import torch

from PiesArt import StimArtifactDataset
from PiesArt.ArtifactEraser.model import GRU_Denoiser
from torch.utils.data import DataLoader
from mef_tools.io import MefReader




"""
Dat = StimArtifactDataset(1, ['MultiCenteriEEG_physiology', 'MultiCenteriEEG_pathology'], ['RCS'])
DLoad = DataLoader(Dat, batch_size=128, num_workers=8, shuffle=True)
model = GRU_Denoiser()


self = model
self.cuda(7)
optimizer = torch.optim.Adam(self.parameters(), lr=0.0003)


Dat._len = 128*20000


for k, (x_orig, x_art, y_art) in enumerate(DLoad):
    optimizer.zero_grad()

    x_orig = x_orig.float().detach().to(self.device)
    x_art = x_art.float().detach().to(self.device)
    y_art = y_art.float().detach().to(self.device)

    x_rec, att = self(x_art)
    #break

    # x_orig_fft = torch.fft.fft(x_orig, dim=-1).abs() / x_rec.shape[-1]
    # x_rec_fft = torch.fft.fft(x_rec, dim=-1).abs() / x_rec.shape[-1]

    x_orig_fft = torch.stft(x_orig.squeeze(1), n_fft=100, hop_length=50)
    x_rec_fft = torch.stft(x_rec.squeeze(1), n_fft=100, hop_length=50)

    loss_signal = F.mse_loss(x_rec, x_orig) * 10
    #loss_att = F.mse_loss(att, y_art)
    loss_att = F.binary_cross_entropy(att, y_art)# * 0.1
    loss_fft = F.mse_loss(x_rec_fft, x_orig_fft) * 1
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
"""

from PiesArt.ArtifactEraser import configs_ArtifactEraser, models_ArtifactEraser
from PiesArt.ArtifactEraser.trainer import Trainer
from PiesGen.DCGAN import models_DCGAN

cfg = configs_ArtifactEraser['RCS_ArtifactEraser_100k']
self = Trainer(cfg)
self.train()

#
#
# from PiesPro.signal import PSD
#
# model = models_ArtifactEraser['RCS_ArtifactEraser_100k']
# model.eval().cpu()
# Rdr = MefReader('/mnt/Helium/filip/Projects/2021_Sleep_ERP_GS/M5/data/M5_Day_0_2Hz/ieeg M5_Day_0_2Hz.mefd')
# #Rdr = MefReader('/mnt/Helium/filip/Projects/2021_Sleep_ERP_GS/M1/data/M1_Day_2_2Hz/ieeg M1_Day_2_2Hz.mefd')
#
# s = Rdr.get_property('start_time')[0]
# x_real = Rdr.get_data('e12-e15', s, s +60*10*1e6 )
#
# var = np.nanstd(x_real)
# mu = np.nanmean(x_real)
#
# x_real = (x_real - mu) / var
# x_real[np.isnan(x_real)] = 0
#
# x_rec, yy = model(torch.tensor(x_real).float().view(1,1,-1))
#
# plt.plot(x_real)
# plt.plot(x_rec.squeeze().detach().numpy())
# plt.xlim([0, 2000])
# plt.show()
#
#
# plt.plot(yy.squeeze().detach().numpy())
# plt.xlim([0, 2000])
# plt.show()
#
# plt.plot(x_real)
# plt.plot(x_rec.squeeze().detach().numpy())
# plt.xlim([0, 1000])
# plt.show()
#
#
# plt.plot(yy.squeeze().detach().numpy())
# plt.xlim([0, 1000])
# plt.show()
#
#
#
# plt.plot(x_real)
# plt.plot(x_rec.squeeze().detach().numpy())
# plt.xlim([0, 500])
# plt.show()
#
#
# plt.plot(yy.squeeze().detach().numpy())
# plt.xlim([0, 500])
# plt.show()
#
#
# plt.plot(x_real)
# plt.plot(x_rec.squeeze().detach().numpy())
# plt.xlim([4000, 5000])
# plt.show()
#
#
# plt.plot(yy.squeeze().detach().numpy())
# plt.xlim([4000, 5000])
# plt.show()
#



#x_fft_orig = np.abs(np.fft.fft(x_real)) / x_real.shape[0]
#x_fft_orig = x_fft_orig[:int(np.floor(x_fft_orig.shape[0]/2))]

# x_fft_rec = np.abs(np.fft.fft(x_rec.squeeze().detach().numpy())) / x_rec.shape[-1]
# x_fft_rec = x_fft_rec[:int(np.floor(x_fft_rec.shape[0]/2))]
#
# freq = np.linspace(0, 1, x_fft_orig.shape[0]) * 250

#
#
# freq, Pxx_real = PSD(x_real, fs=500, nperseg=5*500, noverlap=5*250)
# freq, Pxx_rec = PSD(x_rec.squeeze().detach().numpy(), fs=500, nperseg=5*500, noverlap=5*250)
# freq, Pxx_sub = PSD(x_real-x_rec.squeeze().detach().numpy(), fs=500, nperseg=5*500, noverlap=5*250)
#
# plt.semilogy(freq, Pxx_real)
# plt.semilogy(freq, Pxx_rec)
# plt.xlim([0.5, 100])
# #plt.ylim([1e-6, 1e0])
# plt.show()
#
# plt.semilogy(freq, Pxx_sub)
# plt.xlim([0.5, 100])
# #plt.ylim([1e-6, 1e0])
# plt.show()
#
#
#
# #
#
# from PiesArt.ArtifactEraser.dataset import StimArtifactDataset
# sig_gen = models_DCGAN['MultiCenteriEEG_pathology']
# sart = StimArtifactDataset()
# sart._len = 100
#
#
# x_orig, x_art, y_ = sart[0]
# x_rec, yy = model(torch.tensor(x_art).float().view(1,1,-1))
#
# freq, Pxx_orig = PSD(x_orig.squeeze(), fs=500, nperseg=2*500, noverlap=2*250)
# freq, Pxx_art = PSD(x_art.squeeze(), fs=500, nperseg=2*500, noverlap=2*250)
# freq, Pxx_rec = PSD(x_rec.squeeze().detach().numpy(), fs=500, nperseg=2*500, noverlap=2*250)
#
#
# plt.plot(x_orig)
# plt.plot(x_art.squeeze().detach().numpy())
# plt.plot(x_rec.squeeze().detach().numpy())
# plt.xlim([4000, 5000])
# plt.show()
#
#
#
# plt.semilogy(freq, Pxx_orig)
# plt.semilogy(freq, Pxx_art)
# plt.semilogy(freq, Pxx_rec)
# plt.xlim([0.5, 200])
# #plt.ylim([1e-6, 1e0])
# plt.show()







