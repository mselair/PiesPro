import torch
import torch.nn as nn
import torch.nn.functional as F

from PiesDL.models_generic import ResLayer1D


class GRU_Denoiser(nn.Module):
    def __init__(self, n_filters=64):
        super().__init__()
        self.dummy_par = nn.Parameter(torch.zeros(1))

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=11, stride=2, padding=5, bias=False)
        self.resl1 = ResLayer1D(n_filters)


        self.gru = nn.GRU(n_filters, hidden_size=n_filters, num_layers=1, bidirectional=False, bias=False, batch_first=True)

        self.gru_attention = nn.GRU(n_filters, hidden_size=n_filters, num_layers=1, bidirectional=False, bias=False, batch_first=True)
        self.fc_attention = nn.Linear(n_filters, 1, bias=True)

        self.conv2 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters, kernel_size=5, stride=1, padding=2, bias=False)
        self.resl2 = ResLayer1D(n_filters, bias=False)

        self.convoutp = nn.ConvTranspose1d(in_channels=n_filters, out_channels=1, kernel_size=12, stride=2, padding=5, bias=False)
        self.convfilter = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=11, stride=1, padding=5, bias=False)

        self.batch_outpatt = nn.BatchNorm1d(1)
        self.convoutp_att = nn.ConvTranspose1d(in_channels=1, out_channels=1, kernel_size=12, stride=2, padding=5, bias=False)



    def forward(self, x_inp):
        # x_inp = x_art

        x = x_inp
        x = self.conv1(x)
        x = F.relu(x)
        x1 = x

        x, _ = self.gru(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        x = self.resl1(x)

        #x_att = torch.sigmoid(self.conv2_attention(x))
        x_att = torch.sigmoid(self.fc_attention(self.gru_attention(x.permute(0, 2, 1))[0])).permute(0, 2, 1)

        x_ = torch.cat((x, x1*x_att), dim=1)
        x_ = F.relu(x_)
        x_ = self.conv2(x_)
        x_ = F.relu(x_)

        x_ = self.resl2(x_)
        x_outp = self.convoutp(x_)
        x_outp = self.convfilter(x_outp)

        x_att = torch.sigmoid(self.batch_outpatt(self.convoutp_att(x_att)))


        return x_outp, x_att


    @property
    def device(self):
        return self.dummy_par.device
