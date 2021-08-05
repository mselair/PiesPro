from PiesPro import DELIMITER
from PiesPro._config import get_files
from PiesArt.ArtifactEraser._configs import configs_ArtifactEraser
from PiesArt.ArtifactEraser.model import GRU_Denoiser
import torch


class _models_ArtifactEraser(dict):

    _keys = dict([
        (
            '_'.join(f.split(DELIMITER)[-1].split('_')[:3]),
            f
        )
        for f in get_files(DELIMITER.join(__file__.split(DELIMITER)[:-1]), 'pt')
        if not '.-' in f.split(DELIMITER)[-1]
    ])

    def keys(self):
        return self._keys.keys()

    def get_model(self, item):
        f = self._keys[item]
        cfg = configs_ArtifactEraser[item]
        state_dict = torch.load(f, map_location='cpu')
        mod = GRU_Denoiser(n_filters = cfg.MODEL.ARCHITECTURE.N_FILTERS)
        mod.load_state_dict(state_dict, strict=True)
        mod.eval()
        return mod

    def __getitem__(self, item):
        return self.get_model(item)

    def __call__(self, item):
        return self[item]

    def __str__(self):
        return 'models_ArtifactEraser: ' + str(self._keys.keys())

    def __repr__(self):
        return self.__str__()

models_ArtifactEraser = _models_ArtifactEraser()






