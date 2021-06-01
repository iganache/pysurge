#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 15:33:21 2021

@author: indujaa
"""
import argparse
from DPDC2D_iso import InputParams, RasterGrid
from DPDC2D_iso import DiluteCurrentModel


def getInput():
    """ 
    Overrides the values input through the config file with command line input 
    '"""
    
    parser = argparse.ArgumentParser(description="Input paramters")
    # parser.add_argument('--config', dest = 'config', required=True, help = 'Path to config file')
    parser.add_argument('--config', dest = 'config', required=True, help = 'Absolute path of config file')
    parser.add_argument('--dem', dest = 'dem', required=True, help = 'Absolute path of DEM file')
    
    return parser.parse_args()


def main():
    args = getInput()

    in_config = InputParams(args.config)
    config_params = in_config.get_params()
    
    # # change dem input style
    dem = RasterGrid(args.dem)
    
    dpdc = DiluteCurrentModel(config_params)
    dpdc.run_model()
    
if __name__ == '__main__':
    main() 
    

