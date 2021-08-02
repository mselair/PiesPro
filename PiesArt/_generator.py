import numpy as np
from PiesArt import stimulation_artifacts

class ArtifactGenerator:
    def __init__(self, artifact_key='RCS', target_fs=500):
        tmp = stimulation_artifacts[artifact_key]
        self.artifact = tmp['X']
        self.fs_artifact = tmp['fs']
        self.fs_target = target_fs
        self.artifact_key = artifact_key

    def get_signal(self, n_batch, n_length, prob=None):
        sig_gen = []
        ref_gen = []

        for n in range(n_batch):
            if prob is None:
                _prob = np.random.rand()
            else:
                _prob = prob

            x = np.zeros(n_length)
            y = np.zeros(n_length)

            s = 0

            x_art = self.get_artifact()
            while s < x.shape[0] - 1 - x_art.shape[0] - x_art.shape[-1]-1:
                if np.random.rand() > (1 - _prob):
                    x_art = self.get_artifact()
                    s = s + np.random.randint(x_art.shape[0])
                    x[s:s + x_art.shape[0]] += x_art
                    y[s:s + x_art.shape[0]] = 1
                s = s + x_art.shape[0] + 3

            sig_gen += [x]
            ref_gen += [y]
        return np.stack(sig_gen).reshape(n_batch, 1, n_length), np.stack(ref_gen).reshape(n_batch, 1, n_length)

    def get_artifact(self):
        xtemp = self.artifact
        len = int(np.round(self.fs_artifact / self.fs_target))

        s = np.random.randint(len)
        positions = np.arange(s, self.artifact.shape[0], len).astype(int)
        positions_m = (positions + np.random.randint(-len, len, positions.shape[0])).astype(int)
        positions_m[positions_m < 0] = 0
        positions_m[positions_m >= self.artifact.shape[0]] = self.artifact.shape[0] - 1

        temp = xtemp[positions_m]
        if np.random.randn() < 0:
            temp = temp * -1

        # f = np.random.rand() * 10 + 0.5
        f = 10 ** (np.random.rand() * (1.6 - (-1)) - 2)
        temp = temp * f
        if temp.shape[0] == 0: temp = np.append(temp[0])
        return temp
