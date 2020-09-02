# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 04:55:05 2020

@author: Anshul Arya
"""

# Function to add figure number
def add_fignum(caption):
    figtext_args = (0.5, -0.2, caption) 
  
    figtext_kwargs = dict(horizontalalignment ="center",  
                          fontsize = 14, color ="black",
                          wrap = True)
    return figtext_args, figtext_kwargs