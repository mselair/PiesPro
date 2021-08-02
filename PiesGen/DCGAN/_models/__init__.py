from PiesPro import DELIMITER
from PiesPro._config import get_files
from PiesGen.DCGAN._configs import configs_DCGAN
from PiesGen.DCGAN.model import Generator
import torch


class _models_DCGAN(dict):

    _keys = dict([
        (
            '_'.join(f.split(DELIMITER)[-1].split('_')[:2]),
            f
        )
        for f in get_files(DELIMITER.join(__file__.split(DELIMITER)[:-1]), 'pt')
        if not '.-' in f.split(DELIMITER)[-1]
    ])

    def keys(self):
        return self._keys.keys()

    def get_model(self, item):
        f = self._keys[item]
        cfg = configs_DCGAN[item]
        state_dict = torch.load(f, map_location='cpu')
        mod = Generator(n_features = cfg.MODEL.ARCHITECTURE.N_EMBEDDED_FEATURES, n_filters = cfg.MODEL.ARCHITECTURE.N_FILTERS)
        mod.load_state_dict(state_dict, strict=True)
        mod.eval()
        return mod

    def __getitem__(self, item):
        return self.get_model(item)

    def __call__(self, item):
        return self[item]

    def __str__(self):
        return 'models_DCGAN: ' + str(self._keys.keys())

    def __repr__(self):
        return self.__str__()

models_DCGAN = _models_DCGAN()





