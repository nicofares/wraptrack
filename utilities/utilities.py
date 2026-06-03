# -*- coding: utf-8 -*-

"""
Wrapper: utilities for frame modification, video saving and conversion, etc. 

Author: nicofares

Creation: 2026-05-29
"""

import numpy as np

def remove_background(img, bg):
    return img - bg

def normalize8(I):
    mn = I.min()
    mx = I.max()
    I = ((I - mn) / (mx - mn)) * 255
    return I.astype(np.uint8)

def vid2frames(vidpath, savepath='./temp/', vidformat='mp4', frameformat='png'):
    pass

def frames2vid(framespath, savepath='./temp', frameformat='png', vidformat='mp4'):
    pass

def cropvid():
    pass