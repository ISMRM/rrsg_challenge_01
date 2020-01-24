#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 12:05:59 2018

@author: omaier

Copyright 2019 Oliver Maier

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

"""
import numpy as np


def cmp(k, cmp_type=None):
    if len(np.shape(k)) == 2:
        nspokes, N = np.shape(k)
    elif len(np.shape(k)) == 3:
        NScan, nspokes, N = np.shape(k)
    else:
        raise ValueError("Passed trajectory has the wrong "
                         "number of dumensions.")

    w = np.abs(np.linspace(-N/2, N/2, N))/(N/2)  # ramp from -1...1
    w = np.repeat(w, nspokes, 0)
    # w /= np.min(w)
    w *= (np.pi / 4) / nspokes
    w = np.reshape(w, (N, nspokes)).T
    return w
