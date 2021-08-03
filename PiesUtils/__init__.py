# Copyright 2020-present, Mayo Clinic Department of Neurology - Laboratory of Bioelectronics Neurophysiology and Engineering
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#import PiesSleep.PiesUtils.feature
#import PiesSleep.PiesUtils.cv2_util
#import PiesSleep.PiesUtils.files
#import PiesSleep.PiesUtils.signal
#import PiesSleep.PiesUtils.stats
#import PiesSleep.PiesUtils.types



"""

A Python package providing tools and utilities which are nice to have while working not only on this project.

"""

#
#
# """
# feature_util
# ^^^^^^^^^^^^^^^^
#
# Tools utilized in machine learning pipelines for feature augmentation, outlier detection, z-score normalization, balancing classes etc.
#
#
# signal
# ^^^^^^^^^^^^^^^^
#
# Tools for digital signal processing: fft filtering, low-frequency filtering utilizing downsampling etc.
#
#
# types
# ^^^^^^^^^^^^^^^^
#
# * ObjDict
#     - Combines Matlab struct and python dict. Mirroring keys and attributes and its values. Nested initialization implemented as well.
#
#
# .. code-block:: python
#
#     from PiesSleep.PiesUtils.types import ObjDict
#
#     x = ObjDict()
#     x['a'] = 1
#     x.b = 2
#     print(x['a'], x.a)
#     print(x['b'], x.b)
#
#     del x.b
#     print(x)
#     del x['a']
#
#     x.a.b.c = 10 # creates nested ObjDicts
#     x['b']['b']['c'] = 20
#
#
#
# """
