# -*- coding: utf-8 -*-

import sys
import os
import argparse
from pathlib import Path
import configparser
import numpy as np
# import math
from scipy import special
import matplotlib.pyplot as plt
#from matplotlib.colors import LogNorm
import rasterio


class RasterGrid():
    """
    A class to represent all georeferenced input and output raster dataset.
    This includes DEM files, and output files.
    
    Attributes
    ----------
    DEMfile : str
        absolute path to the input DEM file
        
    Methods
    -------
    write_tiff(arr, outfile, nodata, count, minvalue = 1e-6)
        writes numpy arrays into geotiffs using rasterio
    """
    
    def __init__(self, DEMfile):

        ds = rasterio.open(DEMfile)
        self.ny = ds.height
        self.nx = ds.width
        self.crs = ds.crs
        self.transform = ds.transform
        self.dx = ds.transform[0]
        self.dy = ds.transform[4]
        self.elev = ds.read(1)
        ds.close()
        
    def write_tiff(self, arr, outfile, nodata, count, minvalue = 1e-6):
        
        w_arr = arr.copy()
        if nodata != None:
            w_arr[np.abs(w_arr)<minvalue] = nodata

        outds = rasterio.open(outfile, 'w', driver='GTiff', 
                          height = self.ny, 
                          width = self.nx, 
                          count=count, 
                          crs = self.crs, 
                          dtype = arr.dtype,
                          transform = self.transform,
                          nodata = nodata)

        if w_arr.ndim > 2 and count == w_arr.shape[0]:
            for i in range(count):
                outds.write(w_arr[i], i+1)
        elif w_arr.ndim == 2 and count == 1:      
            outds.write(w_arr, count)

        outds.close()
        
        
        
class NoCRSGrid():
    """
    A class to represent all input and output data that aren't georeferenced.
    
    Attributes
    ----------
    DEMfile : str
        absolute path to the input DEM file
    nx: int
        number of grid cells in the x direction
    nx: int or None
        number of grid cells in the y direction
        
    Methods
    -------
    write_tiff(arr, outfile, nodata, count, minvalue = 1e-6)
    write_tiff1D(arr, outfile, nodata, count, minvalue = 1e-6)
    write_txt1D(arr, outfile, nodata, count, minvalue = 1e-6)
        
    """
    
    def __init__(self, DEMfile, nx, ny=1):
        data = np.loadtxt(DEMfile).reshape(ny,nx)
        self.ny = ny
        self.nx = nx
        self.crs = None
        self.transform = None
        self.elev = data
        self.dx = 1
        self.dy = 1
        
        
    def write_tiff(self, arr, outfile, nodata, count, minvalue = 1e-6):
        if nodata != None:
            arr[np.abs(arr)<minvalue] = nodata

        outds = rasterio.open(outfile, 'w', driver='GTiff', 
                          height = self.ny, 
                          width = self.nx, 
                          count=count, 
                          crs = self.crs, 
                          dtype = arr.dtype,
                          transform = self.transform,
                          nodata = nodata)

        if arr.ndim > 2 and count == arr.shape[0]:
            for i in range(count):
                outds.write(arr[i], i+1)
        elif arr.ndim == 2 and count == 1:      
            outds.write(arr, count)

        outds.close()
        
        
    def write_tiff1D(self, arr, outfile, nodata, count, minvalue = 1e-6):
        if nodata != None:
            arr[np.abs(arr)<minvalue] = nodata

        outds = rasterio.open(outfile, 'w', driver='GTiff', 
                          height = self.ny, 
                          width = self.nx, 
                          count=count, 
                          crs = self.crs, 
                          dtype = arr.dtype,
                          transform = self.transform,
                          nodata = nodata)

        if arr.ndim > 1 and count == arr.shape[0]:
            arr=arr.reshape((count, self.ny, self.nx))
            for i in range(count):
                outds.write(arr[i], i+1)
        elif arr.ndim == 1 and count == 1: 
            arr=arr.reshape((self.ny, self.nx))
            outds.write(arr, count)

        outds.close()
        
    def write_txt1D(self, arr, outfile, nodata, count, minvalue = 1e-6):
        if nodata != None:
            arr[np.abs(arr)<minvalue] = nodata
            
        # # rewrite to include time snap in filename
        if arr.ndim > 1 and count == arr.shape[0]:
            np.savetxt(outfile, arr)
        elif arr.ndim == 1 and count == 1: 
             np.savetxt(outfile, arr)
        
