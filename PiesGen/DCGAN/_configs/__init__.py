from PiesPro import DELIMITER
from PiesPro._config import get_files, config

configs_DCGAN = dict([(f.split(DELIMITER)[-1].split('.')[0], config(f)) for f in get_files(DELIMITER.join(__file__.split(DELIMITER)[:-1]), 'yaml') if not '.-' in f.split(DELIMITER)[-1]])




