import torch
from PiesGen.DCGAN._models import models_DCGAN
from PiesArt import ArtifactGenerator

class StimArtifactDataset:
    def __init__(self, sig_len=60 ,use_models=['MultiCenteriEEG_pathology', 'MultiCenteriEEG_physiology'], use_artifacts=['RCS'], device='cpu'):
        self._device = device
        self.SigGenerators = [models_DCGAN[k].to(device) for k in use_models]
        self.ArtGenerators = [ArtifactGenerator(k, 500) for k in use_artifacts]
        self._len = 1000000
        self.sig_len = sig_len


    def to(self, device):
        self.SigGenerators = [sg.to(device) for sg in self.SigGenerators]

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self.to(device)

    def __getitem__(self, item):
        idx_sig_gen = item % self.SigGenerators.__len__()
        idx_art_gen = item % self.ArtGenerators.__len__()

        X_orig = self.SigGenerators[idx_sig_gen].generate_signal(n_batch=1, n_seconds=self.sig_len, momentum=0.1).squeeze(0)
        X_art, Y_art = self.ArtGenerators[idx_art_gen].get_signal(n_batch=1, n_length=X_orig.shape[-1])
        X_art = torch.tensor(X_art, dtype=torch.float32).squeeze(0)
        Y_art = torch.tensor(Y_art, dtype=torch.float32).squeeze(0)
        X_art = X_orig + X_art
        return X_orig, X_art, Y_art

    def __len__(self):
        return self._len







