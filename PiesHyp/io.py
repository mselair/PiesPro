# Copyright 2020-present, Mayo Clinic Department of Neurology
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
from io import StringIO
import xml.etree.ElementTree as ET
import os
import pandas as pd

from io import StringIO
import xml.etree.ElementTree as ET
import os
from tqdm import tqdm

from mef_tools.io import MefReader

from PiesHyp.CyberPSG import CyberPSGFile, CyberPSG_XML_Writter
from PiesHyp.NSRR import NSRRSleepFile
from PiesHyp.utils import time_to_utc, create_duration, tile_annotations
import pandas as pd


"""
Tools for loading and saving of PiesHyp files.
"""

_hypnogram_colors = {
    'AWAKE': '#e7b233',
    'WAKE': '#e7b233',
    'Arousal': '#d44b05',
    'REM': '#3500d3',
    'N1': '#2bc7c4',  # 2b7cc7
    'N2': '#2b5dc7',
    'N3': '#000000',
    'SLP': '#2b5dc7',
    'UNKNOWN': '#eaeded',
    'C_N2Huijie_1154_19_07_04': '#FF548ED5',
    'C_AWAKEHuijie_1154_19_07_04': '#FFFFFF00',
    'C_REMHuijie_1154_19_07_04': '#FF7030A0',
    'AWAKE_b': '#e7b233',
    'REM_b': '#3500d3',
    'N1_b': '#2bc7c4',  # 2b7cc7
    'N2_b': '#2b5dc7',
    'N3_b': '#000000',
    'AWAKE_dr': '#e7b233',
    'REM_dr': '#3500d3',
    'N1_dr': '#2bc7c4',  # 2b7cc7
    'N2_dr': '#2b5dc7',
    'N3_dr': '#000000',
    'AWAKE_aisc': '#e7b233',
    'REM_aisc': '#3500d3',
    'N1_aisc': '#2bc7c4',  # 2b7cc7
    'N2_aisc': '#2b5dc7',
    'N3_aisc': '#000000',
    'N_aisc': '#000000',
    'SLP_aisc': '#000000',

}


def load_CyberPSG(path, tile=None, verbose=True):
    if isinstance(path, list):
        return _load_CyberPSG_dataset(path, tile, verbose)
    else:
        return _load_CyberPSG(path, tile)


def _load_CyberPSG(path, tile=None):
    if not os.path.isfile(path):
        raise FileNotFoundError('[FILE ERROR]: File not found ' + path)
    fid = CyberPSGFile(path)
    annotations = fid.get_hypnogram()
    df = pd.DataFrame(annotations)
    df = create_duration(df)
    if not isinstance(tile, type(None)):
        if (df.duration > tile).sum() > 0:
            df = tile_annotations(df, tile)

    for k in df.annotation.unique():
        if k[-5:] == '_aisc':
            df.loc[df.annotation == k, 'annotation'] = k[:-5]
    return df


def _load_CyberPSG_dataset(paths: list, tile=None, verbose=True):
    if verbose:
        print('Loading Hypnogram Dataset')
        return [_load_CyberPSG(pth, tile) for pth in tqdm(paths)]
    else:
        return [_load_CyberPSG(pth, tile) for pth in paths]



def save_CyberPSG(path, df):
    #TODO: Do Tests
    #TODO: Implement annotation groups etc

    fid = CyberPSG_XML_Writter(path)
    annotation_group = 'Import_aisc'
    annotation_types = list(df['annotation'].unique())
    df = time_to_utc(df)

    fid.add_AnnotationGroup(annotation_group, uuid_='00000000-0000-0000-0000-000000000001')
    for atype in annotation_types:
        fid.add_AnnotationType(atype, groupAssociationId=annotation_group, color=_hypnogram_colors[atype])

    for idx, row in df.iterrows():
        fid.add_Annotation(row['start'], row['end'], AnnotationTypeId=row['annotation'])
    fid.dump()


def load_NSRR(path):
    return NSRRSleepFile(path).get_hypnogram()


def load_MefFile(mefid: MefReader):
    hyp = pd.DataFrame(mefid.get_annotations())
    hyp['duration'] = hyp['duration']/1e6
    hyp = hyp.drop(columns='type')

    hyp['start'] = hyp['time']/1e6
    hyp = hyp.drop(['time'], axis=1)

    hyp['end'] = hyp['start'] + hyp['duration']

    hyp['annotation'] = hyp['text']
    hyp = hyp.drop(['text'], axis=1)

    hyp = hyp[['annotation', 'start', 'end', 'duration']]

    return hyp

import numpy as np




