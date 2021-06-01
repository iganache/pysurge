# -*- coding: utf-8 -*-
"""
Created on Fri May  7 02:08:36 2021

@author: Indujaa
"""
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


class InputParams():
    
    def __init__(self, configfile):
        self.infile = configfile        
        
    def get_params(self):
        
        config=configparser.ConfigParser()
        config.read(self.infile)

        output = config['path']

        # Simulation parameters
        sim_params=config['Simulation_params']
        
        # Physical parameters
        phys_params=config['Physical_params']
        
        # Atmospheric parameters
        atmo_params=config['Atmo_params']
        
        # Dilute current parameters
        volc_params=config['Volc_params']
        
        # DEM parameters
        DEM_params=config['DEM']
        

        params = dict(outpath = output['out_path'],
                      outdir = output['out_dir'],
                      DEM = DEM_params['dem_tiff'],
                      total_time = np.int(sim_params['total_time']),
                      num_vents = np.int(sim_params['num_vents']),
                      num_frames = np.int(sim_params['num_frames']),
                      timestep = np.float(sim_params['timestep']),
                      cstable = np.float(sim_params['cstable']),
                      maxstep = np.float(sim_params['maxstep']),
                      minstep = np.float(sim_params['minstep']),
                      order = np.int(sim_params['order']),
                      gravity = np.float(phys_params['gravity']),
                      rho_air = np.float(atmo_params['rho_air']), 
                      mu_air = np.float(atmo_params['mu_air']),
                      Cp_air = np.float(atmo_params['Cp_air']),
                      rho_gas = np.float(volc_params['rho_gas']), 
                      n_gas = np.float(volc_params['n_gas']),
                      mu_gas = np.float(volc_params['mu_gas']),
                      Cp_gas = np.float(volc_params['Cp_gas']),
                      rho_dep = np.float(volc_params['rho_deposit']),
                      rho_s = np.float(volc_params['rho_solid']), 
                      n_s = np.float(volc_params['n_solid']),
                      C_s = np.float(volc_params['C_solid']),
                      d50 = np.float(volc_params['d50']),
                      Cd = np.float(volc_params['Cd']),
                      pile_ht = np.float(volc_params['height']),
                      centerXgeo = str(volc_params['centerXgeo']),
                      centerYgeo = str(volc_params['centerYgeo']),
                      vmag = np.float(volc_params['velocity']),
                      vdir = np.deg2rad(np.float(volc_params['direction'])))
                      

        params['hfilm'] = 1e-5
        
        return params

class RasterGrid():
    
    def __init__(self, DEMfile):

        ds = rasterio.open(DEMfile)
        self.hight = ds.height
        self.width = ds.width
        self.crs = ds.crs
        self.transform = ds.transform
        self.dx = ds.transform[0]
        self.dy = ds.transform[4]
        self.elev = ds.read(1)
        ds.close()
        
    def write_tiff(self, arr, outfile, nodata, count, minvalue = 1e-6):
        if nodata != None:
            arr[np.abs(arr)<minvalue] = nodata

        outds = rasterio.open(outfile, 'w', driver='GTiff', 
                          height = self.height, 
                          width = self.width, 
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
    

class DiluteCurrentModel():
    
    def __init__(self, inparams):
        # # Setting model parameters
        self.tsim = inparams['total_time']
        self.dt = inparams['timestep']
        self.cstable = inparams['cstable']
        self.maxdt = inparams['maxstep']
        self.mindt = inparams['minstep']
        self.order = inparams['order']
        
        self.numframes = inparams['num_frames']
        
        self.hfilm = inparams['hfilm']
        
        self.g = inparams['gravity']
        
        self.rho_a = inparams['rho_air']
        self.mu_a = inparams['mu_air']
        self.Cp_a = inparams['Cp_air']
        
        self.rho_s = inparams['rho_s']
        self.phi_s0 = inparams['n_s']
        self.C_s = inparams['C_s']
        
        self.rho_dep = inparams['rho_dep']
        
        self.rho_g = inparams['rho_gas']
        self.phi_g = inparams['n_gas']
        self.mu_g = inparams['mu_gas']
        self.Cp_g = inparams['Cp_gas']

        
        self.d50 = inparams['d50']
        self.Cdrag = inparams['Cd']
        
        self.h = inparams['height']
        self.vel = inparams['velocity']
        self.dir = inparams['direction']
        self.nx = grid.width
        self.ny = grid.height
        self.dx = grid.dx
        self.topo = grid.elev
        
        # # Initializing fields
        # # Need to add stuff for thermal and density
        self.nn = 4
        self.h = self.hfilm + np.zeros((self.ny, self.nx), dtype = np.float64)
        self.h_dep = self.hfilm + np.zeros((self.ny, self.nx), dtype = np.float64)
        self.mass = np.zeros((self.ny, self.nx), dtype = np.float64)
        self.velx = np.zeros((self.ny, self.nx), dtype = np.float64)
        self.vely = np.zeros((self.ny, self.nx), dtype = np.float64)
        self.vel = np.zeros((self.ny, self.nx), dtype = np.float64)
        self.momx = np.zeros((self.ny, self.nx), dtype = np.float64)
        self.momy = np.zeros((self.ny, self.nx), dtype = np.float64)
        self.rho = np.zeros((self.ny, self.nx), dtype = np.float64)
        self.phi_s = np.zeros((self.ny, self.nx), dtype = np.float64)
        
        # # Inititalizing secondary variables
        self.slope = np.zeros((self.ny, self.nx), dtype = np.float64)
        self.slopex = np.zeros((self.ny, self.nx), dtype = np.float64)
        self.slopey = np.zeros((self.ny, self.nx), dtype = np.float64)
        self.gx = np.zeros((self.ny, self.nx), dtype = np.float64)
        self.gy = np.zeros((self.ny, self.nx), dtype = np.float64)
        self.gz = np.zeros((self.ny, self.nx), dtype = np.float64)
        
        self.maxvel = np.zeros((self.ny, self.nx), dtype = np.float64)
        self.maxdepth = np.zeros((self.ny, self.nx), dtype = np.float64)
        
        # # Output files
        self.outfiles_dir = self.set_output_directory(inparams['out_path'], inparams['out_dir'])
        
    def set_output_directory(self, path, newdir):
        outpath = Path(path) / newdir
        outpath_str = str(outpath)

        # outpath = Path(outdir)
        if not outpath.exists():
            os.makedirs(outpath_str)
        elif outpath.exists():
            [f.unlink() for f in outpath.glob('*') if f.is_file()]

        outpath_str = outpath_str+'/'
        return outpath_str

    def make_ghostcells(self, arrays, const):
        """ 
        Pads the input arrays with the input constant an all sides and returns the padded array 
        """ 

        if type(arrays) == list:
            arraylist = []
            for arraynd in arrays:
                if arraynd.ndim == 2:
                    arraynd = np.pad(arraynd, ((1,1), (1,1)), mode='constant', constant_values=((const,const),(const,const)))
                    arraylist.append(arraynd)
                elif arraynd.ndim == 3:
                    arraynd = np.pad(arraynd, ((0,0), (1,1), (1,1)), mode='constant', constant_values=((const,const),(const,const),(const,const)))
                    arraylist.append(arraynd)
            return arraylist 

        elif type(arrays) == np.ndarray:
            arrays =  np.pad(arrays, ((1,1), (1,1)), mode='constant', constant_values=((const,const),(const,const)))
            return arrays
        
        
    def n_a(self):
        return 1 - self.n_g - self.n_s
    
    def vel_total(self, velx=0, vely=0):
        if velx == 0:
            velx=self.velx 
            vely=self.vely
        return np.sqrt(velx**2 + vely**2)
    
    
    def set_initial_cond(self):
        # # Recalculating air density based on pre layer thickness h_film
        self.rho_a = self.hfilm * self.rho_s + (1-self.hfilm) * self.rho_a
        
        ### Intitalizing conservative variables ######################
          
        for i in range(len(Flow['centerX'])):
            self.h[Flow['centerY'][i], Flow['centerX'][i]] = self.height
#             self.h_dep = self.h
            self.rho[Flow['centerY'][i], Flow['centerX'][i]] =  self.phi_s0*self.rho_s + (1-self.phi_s0)*self.rho_g
            self.phi_s[Flow['centerY'][i], Flow['centerX'][i]] =  phi_s
            self.mass[Flow['centerY'][i], Flow['centerX'][i]] = self.rho * self.height
            self.velx[Flow['centerY'][i], Flow['centerX'][i]] = self.velocity * np.cos(self.dir)
            self.vely[Flow['centerY'][i], Flow['centerX'][i]] = self.velocity * np.sin(self.dir)
            self.vel[Flow['centerY'][i], Flow['centerX'][i]] = self.velocity

            
        self.momx = self.mass * self.velx
        self.momy = self.mass * self.vely
        
        self.topo, self.h, self.mass, self.momx, self.momy, self.rho, self.phi_s, self.velx, self.vely, self.vel, self.maxvel, self.maxdepth = \
        self.make_ghostcells([self.topo, self.h, self.mass, self.momx, self.momy, self.rho, self.phi_s, self.velx, self.vely, self.vel, self.maxvel, self.maxdepth])
            
        self.u = np.array([self.h, self.mass, self.momx, self.momy])


        #### Set boundary conditions for topography #############
        ### Upper ###
        self.topo[0,:] = self.topo [1,:]

        ### Lower ###
        self.topo[-1,:] = self.topo [-2,:]

        ### Left ###
        self.topo[:,0] = self.topo[:,1]

        ### Right ###
        self.topo[:,-1] = self.topo[:,-2]

        self.slopex = np.gradient(self.topo, self.dx, axis = 1)                    # Slope in x direction 
        self.slopey = np.gradient(self.topo, self.dx, axis = 0)                    # Slope in y direction 
        self.slope = np.sqrt(self.slopex **2 + self.slopey **2 + 1)                # True slope. Adding 1 to avoid division by zero
        self.gy = self.g * self.slopey / self.slope                   # Gravitational forcing in y direction
#         sy = - gy * u[0,:,:]                                        # Sorce term for momentum equation in y
        self.gx = self.g * self.slopex / self.slope                   # Gravitational forcing in x direction
#         sx = - gx * u[0,:,:]
        self.gz = self.g / self.slope                            # Sorce term for momentum equation in x

        self.maxdepth = self.u[0,:,:]
        self.maxvel = self.vel  

    
    def update_boundary_cond(self):
        
        """ Updates BCs at every time step for topo, gx, gy, gz (transmissive / solid not required)
         Updates BCs for conserved quanitites and velocity based on whether the boundary is solid or transmissive """

        LeftSolid = False
        RightSolid = False
        UpperSolid = False
        LowerSolid = False

        ## Left boundary 
        self.gx[:,0] = self.gx[:,1]
        self.gy[:,0] = self.gy[:,1]
        self.gz[:,0] = self.gz[:,1]
        self.vely[:,0] = self.vely[:,1]
        self.u[:,:,0] = self.u[:,:,1]
        if LeftSolid:
            self.velx[:,0] = - self.velx[:,1] 
            self.u[2,:,0] = - self.u[2,:,1]
        else:
            self.velx[:,0] = self.velx[:,1] 


        ## Right boundary
        self.gx[:,-1] = self.gx[:,-2]
        self.gy[:,-1] = self.gy[:,-2]
        self.gz[:,-1] = self.gz[:,-2]
        self.vely[:,-1] = self.vely[:,-2]
        self.u[:,:,-1] = self.u[:,:,-2]
        if RightSolid:
            self.velx[:,-1] = - self.velx[:,-2] 
            self.u[2,:,-1] = - self.u[2,:,-2]
        else:
               self.velx[:,-1] = self.velx[:,-2] 


        ## Upper boundary
        self.gx[0,:] = self.gx[1,:]
        self.gy[0,:] = self.gy[1,:]
        self.gz[0,:] = self.gz[1,:]
        self.velx[0,:] = self.velx[1,:] 
        self.u[:,0,:] = self.u[:,1,:]
        if UpperSolid:
            self.vely[0,:] = - self.vely[1,:]
            self.u[3,0,:] = - self.u[3,1,:]
        else:
            self.vely[0,:] = self.vely[1,:]


        ## Lower boundary
        self.gx[-1,:] = self.gx[-2,:]
        self.gy[-1,:] = self.gy[-2,:]
        self.gz[-1,:] = self.gz[-2,:]
        self.velx[-1,:] = self.velx[-2,:] 
        self.u[:,-1,:] = self.u[:,-2,:]
        if LowerSolid:
            self.vely[-1,:] = - self.vely[-2,:]
            self.u[3,-1,:] = - self.u[3,-2,:]
        else:
            self.vely[-1,:] = self.vely[-2,:]
            
            
    def compute_fluxes(self, u, direction, ul, ur, vl, vr, S_xy, g_z, Smaxtemp, tflux):
    
        ulr = self.MUSCLextrap(direction)                  # ulr shape is nn x 2
        [Smaxtemp, tflux, hstar, ustar] = self.hllc(ulr, ul, ur, vl, vr, S_xy, g_z, direction, Smaxtemp, tflux)

        # [Smaxtemp, tflux, hstar, ustar] = HLLCsolver.hllc(ulr, vl, vr, ul, ur, S_xy, gz, self.hfilm, direction, Smaxtemp, tflux)               ## Cython version

        return Smaxtemp, tflux, hstar, ustar



    def wave_speeds(self, ul, hl, hl_noflow, ur, hr, hr_noflow, ustar, hstar, g_z, Sl, Sr, Sm):
        """ Returns the characteristic wave speed of left, middle and right wave """

        Sl = np.minimum(ul - np.sqrt(g_z*hl), ustar - np.sqrt(g_z*hstar))
        Sm = ustar.copy()
        Sr = np.maximum(ur + np.sqrt(g_z*hr), ustar + np.sqrt(g_z*hstar))

        ### Special cases for dry bed on left / right of cell
        Sl[hl_noflow] = ur[hl_noflow] - (2*np.sqrt(g_z[hl_noflow]*hr[hl_noflow]))
        Sm[hl_noflow] = Sl[hl_noflow]
        Sr[hl_noflow] = ur[hl_noflow] + (np.sqrt(g_z[hl_noflow]*hr[hl_noflow]))

        Sl[hr_noflow] = ul[hr_noflow] - (np.sqrt(g_z[hr_noflow]*hl[hr_noflow]))
        Sr[hr_noflow] = ul[hr_noflow] + (2*np.sqrt(g_z[hr_noflow]*hl[hr_noflow]))
        Sm[hr_noflow] = Sr[hr_noflow]

        return Sl, Sm, Sr   

    def hllc(self, ulr, ul, ur, vl, vr, S_temp, g_z, direction, Smaxtemp, tflux):
        """ Uses HLLC solver to calculate the fluxes at every cell interface (computation is done across the whole grid and not cell by cell).
        Returns the maximum wave speed, temporary flux, hstar and ustar"""

        ## Beware ulr is a 4D array. axis 0, size 2 - conserved quantity at i-1/2 and i+12; axis 1, size 3 - conserved quantities h, hu and hv;
        ## axis 3, size nrows - rows; axis 4, size ncols - values in (row, col)

        n = self.nn
        hl = ulr [0,0,:,:].copy()
        hr = ulr [1,0,:,:].copy() 
        bhl = ulr [0,1,:,:].copy() 
        bhr = ulr [1,1,:,:].copy()

        if direction == 'x':
            buhl = ulr [0,2,:,:].copy() 
            buhr = ulr [1,2,:,:].copy()
            bvhl = ulr [0,3,:,:].copy()
            bvhr = ulr [1,3,:,:].copy()
            mx = nx-1
            my = ny
        elif direction == 'y':
            bvhl = ulr [0,2,:,:].copy() 
            bvhr = ulr [1,2,:,:].copy()
            buhl = ulr [0,3,:,:].copy()
            buhr = ulr [1,3,:,:].copy()
            mx = nx
            my = ny-1


        hstar = np.zeros((my,mx), dtype = np.float64)
        ustar = np.zeros((my,mx), dtype = np.float64)

        Sl = S_temp[0,:,:]
        Sm = S_temp[1,:,:]
        Sr = S_temp[2,:,:]



        hl_flow = hl > self.hfilm
        hl_noflow = hl <= self.hfilm

        hr_flow = hr > self.hfilm
        hr_noflow = hr <= self.hfilm

        ul[hl > self.hfilm] = buhl[hl > self.hfilm] / bhl[hl > self.hfilm]
        vl[hl > self.hfilm] = bvhl[hl > self.hfilm] / bhl[hl > self.hfilm]

        ur[hr > self.hfilm] = buhr[hr > self.hfilm] / bhr[hr >self. hfilm]
        vr[hr > self.hfilm] = bvhr[hr > self.hfilm] / bhr[hr > self.hfilm]


        hstar = (ul + (2 * np.sqrt(g_z*hl)) - ur - (2*np.sqrt(g_z*hr))) ** 2 / (16 * g_z)
        ustar = (0.5 * (ul + ur)) + np.sqrt(g_z*hl) - np.sqrt(g_z*hr)


        #### wave speeds ###################
        Sl, Sm, Sr = self.wave_speeds(ul, hl, hl_noflow, ur, hr, hr_noflow, ustar, hstar, g_z, Sl, Sm, Sr)
        
        
        ### Fluxes #############################
        cond_l = Sl >= 0
        cond_m = (Sl < 0) & (Sr > 0)
        cond_r = Sr <= 0
        
        tflux_buhl = (hl[cond_l] * -self.rho_a * g_z[cond_l] * hl[cond_l]) + (bhl[cond_l] * (-ul[cond_l] ** 2 + 0.5 * g_z[cond_l] * hl[cond_l])) + (buhl[cond_l]  * 2*ul[cond_l])
        tflux_buhl = (hr[cond_r] * -self.rho_a * g_z[cond_r] * hr[cond_r]) + (bhr[cond_r] * (-ur[cond_r] ** 2 + 0.5 * g_z[cond_r] * hr[cond_r])) + (buhr[cond_r]  * 2*ur[cond_r])
        
        tflux[0,:,:][cond_l] = hl[cond_l] * ul[cond_l]
        tflux[1,:,:][cond_l] = buhl[cond_l] 
        tflux[2,:,:][cond_l] = tflux_buhl
        
        tflux[0,:,:][cond_m] = (Sr[cond_m]*(hl[cond_m]*ul[cond_m]) - Sl[cond_m]*(hr[cond_m]*ur[cond_m]) + Sl[cond_m]*Sr[cond_m]*(hr[cond_m]-hl[cond_m])) / (Sr[cond_m]-Sl[cond_m])
        tflux[1,:,:][cond_m] = (Sr[cond_m]*(buhl[cond_m]) - Sl[cond_m]*(buhr[cond_m]) + Sl[cond_m]*Sr[cond_m]*(bhr[cond_m]-bhl[cond_m])) / (Sr[cond_m]-Sl[cond_m])
        tflux[2,:,:][cond_m] = (Sr[cond_m]*tflux_buhl - Sl[cond_m]*tflux_buhr + Sl[cond_m]*Sr[cond_m]*(buhr[cond_m] - buhl[cond_m])) / (Sr[cond_m]-Sl[cond_m])

        tflux[0,:,:][cond_r] = hr[cond_r] * ur[cond_r]
        tflux[1,:,:][cond_r] = buhr[cond_r] 
        tflux[2,:,:][cond_r] = tflux_buhr

        cond2_gt = Sm >= 0
        cond2_lt = Sm < 0
        tflux[3,:,:][cond2_gt] =  buhl[cond2_gt] * vl[cond2_gt]
        tflux[3,:,:][cond2_lt] = buhr[cond2_lt] * vr[cond2_lt]


        ### Maximum wave speeds for stability criteria #####
        Smaxtemp = np.maximum(np.absolute(Sl),np.absolute(Sr), np.absolute(Sm))        
        return Smaxtemp, tflux, hstar, ustar

    def MUSCLextrap(self, direction):

        """ Monotonic Upwind Scheme blah blah something. Returns value at i-1 and i+1 for 1st order (I think) """

        if direction == 'x':
            val = self.u[:,:,:-1]
            valplus = self.u[:,:,1:]


        if direction == 'y':
            val = self.u[:,:-1,:]
            valplus = self.u[:,1:,:]


        return np.array([val, valplus])

    def adaptive_timestep(self, time, Smax):
        self.dt = self.cstable * self.dx / Smax

        if self.dt > self.maxdt:
            self.dt = self.maxdt
        if self.tsim+self.dt > self.tsim:
            self.dt = self.tsim - time
        courant = Smax * self.dt / dx

        return self.dt, courant 
    
    def gravity_term(self, dir):
        if dir == 'x':
            return (self.rho - self.rho_a) * self.gx * self.u[0,:,:]
        elif dir == 'y':
            return (self.rho - self.rho_a) * self.gy * self.u[0,:,:]
        
    def settling_term(self):
        
        ws = np.sqrt((4 * (self.rho_s - self.rho) * self.g * self.d50) / (3 * self.Cd * self.rho))
        # # add velocity while using for the momentum equations
        
        return self.rho_s * ws * self.phi_s

        
    def drag_term(self, dir):
        if dir == 'x':
            return - self.Cd * self.rho * self.velx ** 2
        elif dir == 'y':
            return self.Cd * self.rho * self.vely ** 2
        
    def update_CV(self, CV, fluxx, fluxy, sourceterm=0):
        oneoverdx = 1/self.dx
        CV_new = CV - self.dt * oneoverdx * np.diff(fluxx, axis=-1) - self.dt * oneoverdx * np.diff(fluxy, axis=-2) + self.dt * sourceterm

        return CV_new
    
    def set_tempvariables(self):
        #### Variables for storing shit
    
        h = np.zeros((self.ny, self.nx), dtype = np.float64)

        #### Sourrce terms
        betax = np.zeros((self.ny, self.nx), dtype = np.float64)
        betay = np.zeros((self.ny, self.nx), dtype = np.float64)

        #### Left and right quantities
        ulx = np.zeros((self.ny,self.nx-1), dtype = np.float64)
        urx = np.zeros((self.ny,self.nx-1), dtype = np.float64)
        vlx = np.zeros((self.ny,self.nx-1), dtype = np.float64)
        vrx = np.zeros((self.ny,self.nx-1), dtype = np.float64)

        uly = np.zeros((self.ny-1,self.nx), dtype = np.float64)
        ury = np.zeros((self.ny-1,self.nx), dtype = np.float64)
        vly = np.zeros((self.ny-1,self.nx), dtype = np.float64)
        vry = np.zeros((self.ny-1,self.nx), dtype = np.float64)

        #### Wave speeds
        S_x = np.zeros((self.nn,self.ny,self.nx-1), dtype = np.float64)
        S_y = np.zeros((self.nn,self.ny-1,self.nx), dtype = np.float64)
        Smaxtemp = np.zeros((self.ny, self.nx), dtype = np.float64)

        #### Fluxes
        tfluxx = np.zeros((self.nn,self.ny,self.nx-1), dtype = np.float64)
        tfluxy = np.zeros((self.nn,self.ny-1,self.nx), dtype = np.float64)


        return h, betax, betay, ulx, urx, vlx, vrx, uly, ury, vly, vry, S_x, S_y, Smaxtemp, tfluxx, tfluxy
    
    
    def clearVariables(self, arr_list):
        new_arr_list = []
        for arr in arr_list:
            arr *= 0
            new_arr_list.append(arr)

        return new_arr_list

    
    def create_output_movies(self):
        self.movie_h = np.zeros((self.numframes, self.ny, self.nx), dtype=np.float64)
        self.movie_momx = np.zeros((self.numframes, self.ny, self.nx), dtype=np.float64)
        self.movie_momy = np.zeros((self.numframes, self.ny, self.nx), dtype=np.float64)
        self.movie_mass = np.zeros((self.numframes, self.ny, self.nx), dtype=np.float64)
        self.movie_vel = np.zeros((self.numframes, self.ny, self.nx), dtype=np.float64)
        self.movie_dep = np.zeros((self.numframes, self.ny, self.nx), dtype=np.float64)
        
    def update_output_movies(self):
        self.movie_h = self.u[0,1:-1,1:-1]
        self.movie_momx = self.u[2,1:-1,1:-1]
        self.movie_momy = self.u[3,1:-1,1:-1]
        self.movie_mass = self.u[1,1:-1,1:-1]
        self.movie_vel = self.vel[1:-1,1:-1]
        self.movie_dep = np.zeros((self.numframes, self.ny, self.nx), dtype=np.float64)
    
    def output_variables(self, maxdepth, maxvel):
                
        h = self.u[0,:,:]
        cond = h>self.hfilm
        no_cond = h<=self.hfilm
        
        self.u[0,:,:][no_cond] = self.hfilm
        self.u[1,:,:][no_cond] = 0
        self.u[2,:,:][no_cond] = 0    
        self.u[3,:,:][no_cond] = 0    
        
        # # velocity
        self.velx[cond] = self.u[2,:,:][cond]  / self.u[1,:,:][cond] 
        self.vely[cond] = self.u[3,:,:][cond]  / self.u[1,:,:][cond]
        self.velx[h<=0.01] = 0
        self.vely[h<=0.01] = 0
        self.vel = self.vel_total()
        
        # # current density
        self.rho[cond] = self.u[1,:,:][cond]  / self.u[0,:,:][cond] 
        self.rho[no_cond] = self.rho_a
        self.phi_s[cond] = (self.rho[h>self.hfilm]-self.rho_g) / (self.rho_s - self.rho_g[cond])
        self.phi_s[no_cond] = self.h_film

        self.maxvel = np.maximum(maxvel, self.vel)
        self.maxdepth = np.maximum(maxdepth, self.u[0,:,:])

        return maxdepth, maxvel

    def check_liftoff(self):
        # # ADDRESS: Does it need to be changed to account for only flow front density reversal?
        
        current_cond = self.u[0,:,:]>self.hfilm
        if np.mean(self.rho[current_cond]) >= 0.1*self.rho_a:
            self.t = 2 * self.tsim
    
    def create_output_files(self):
        #### Final results as binary files #########
        grid.write_tiff(self.u[0,1:-1,1:-1], self.outfiles_dir+"Depth.tif", -9999, 1, .001)
        grid.write_tiff(self.u[1,1:-1,1:-1], self.outfiles_dir+"Mass.tif", 0, 1, 0)
        grid.write_tiff(self.u[2,1:-1,1:-1], self.outfiles_dir+"MomentumX.tif", 0, 1, 0)
        grid.write_tiff(self.u[3,1:-1,1:-1], self.outfiles_dir+"MomentumY.tif", 0, 1, 0)
        grid.write_tiff(self.h_dep[1:-1,1:-1], self.outfiles_dir+"Deposit_depth.tif", -9999, 1, .001)
        grid.write_tiff(self.vel[1:-1,1:-1], self.outfiles_dir+"Velocity.tif", 0, 1, .001)
        grid.write_tiff(self.maxdepth[1:-1,1:-1], outdir+"MaxDepth.tif", -9999, 1, 1e-5)
        grid.write_tiff(self.maxvel[1:-1,1:-1], outdir+"MaxVelocity.tif", 0, 1)


        #### Final results as movie files #########
        grid.write_tiff(self.movie_h, self.outfiles_dir+"DepthMovie.tif", -9999, self.numframes, .001)
        grid.write_tiff(self.movie_momx, self.outfiles_dir+"MomentumXMovie.tif", 0, self.numframes, 0)
        grid.write_tiff(self.movie_momy, self.outfiles_dir+"MomentumYMovie.tif", 0, self.numframes, 0)
#         grid.write_tiff(self.movie_velx, self.outfiles_dir+"VelocityXMovie.tif", 0, self.numframes, .001)
#         grid.write_tiff(self.movie_vely, self.outfiles_dir+"VelocityYMovie.tif", 0, self.numframes, .001)
        grid.write_tiff(self.movie_vel, self.outfiles_dir+"VelocityMovie.tif", 0, self.numframes, .001)
        
        # # ADDRESS: REWRITE FOR DEPOSIT THICKNESS

#         ## TIFF file for classification
#         u_bin = u[0,1:-1,1:-1].astype(np.int8)
#         u_bin[u_bin >= 0.1] = 1                                     ## binary raster with depth>20cm = 1 and depth<20cm = 0
#         u_bin[u_bin < 0.1] = 0
#         grid.write_tiff(u_bin, outdir+"Depthbin.tif", None, 1)
        
    
    def run_model(self):
        ##### Set Initial Conditions ######
        self.set_initial_cond()


        ### Arrays to hold movie time series ############################
        movie_idx = 0    
        self.create_output_movies()
        printat = self.tsim / self.numframes 


        h, betax, betay, ulx, urx, vlx, vrx, uly, ury, vly, vry, S_x, S_y, Smaxtemp, tfluxx, tfluxy = self.set_tempvariables()
        ulx, urx, vlx, vrx, uly, ury, vly, vry, tfluxx, tfluxy, betax, betay = self.make_ghostcells([ulx, urx, vlx, vrx, uly, ury, vly, vry, tfluxx, tfluxy, betax, betay], 0)
        
        # # ADDRESS: SOURCE TERMS BETAX AND BETAY ARE DIFFERENT; NEED TO BE ADDED DIFFERENTLY


        t = 0
        while t < self.tsim:
            self.update_boundary_cond()

            Smax = 0

            # # Flux in X-direction 
            [Smaxtemp, tfluxx, hstarx, ustarx] = self.compute_fluxes(u, 'x', ulx, urx, vlx, vrx, S_x, self.gz[:,:-1], Smaxtemp, tfluxx)
            Smax = max(Smax, np.max(Smaxtemp))
            # # Flux in Y-direction 
            [Smaxtemp, tfluxy, hstary, ustary] = self.compute_fluxes(u, 'y', vly, vry, uly, ury, S_y, self.gz[:-1,:], Smaxtemp, tfluxy)
            Smax = max(Smax, np.max(Smaxtemp))

            # # Adaptive time step 
            self.dt, courant = self.adaptive_timestep(t, Smax)   
            if courant > 1:
                print("CFL error at t = ", t)
                t = 2 * self.tsim
            t += self.dt
            
            # # Source terms for momentum
            betax = self.gravity_term('x') - self.settling_term()*self.velx - self.drag_term('x')
            betay = self.gravity_term('y') - self.settling_term()*self.vely - self.drag_term('y')

            # # Time update
            u[0,1:-1,1:-1] = self.update_CV(u[0,1:-1,1:-1], Simulation['dt'], dx, tfluxx[0,1:-1,:], tfluxy[0,:,1:-1])
            u[1,1:-1,1:-1] = self.update_CV(u[1,1:-1,1:-1], Simulation['dt'], dx, tfluxx[1,1:-1,:], tfluxy[1,:,1:-1], sx[1:-1,1:-1], - self.settling_term()[1:-1,1:-1])
            u[2,1:-1,1:-1] = self.update_CV(u[2,1:-1,1:-1], Simulation['dt'], dx, tfluxx[2,1:-1,:], tfluxy[3,:,1:-1], sx[1:-1,1:-1], betax[1:-1,1:-1])
            u[3,1:-1,1:-1] = self.update_CV(u[3,1:-1,1:-1], Simulation['dt'], dx, tfluxx[3,1:-1,:], tfluxy[2,:,1:-1], sy[1:-1,1:-1], betay[1:-1,1:-1])
            maxdepth, maxvel = self.output_variables(self, maxdepth, maxvel)
            
            # # deposit thickness
            # # ADDRESS: should the thickness be calculated based on density condition or hfilm condition
            current_cond = self.rho>self.rho_a
            self.h_dep[current_cond] = self.h_dep - self.dt* (self.settling_term() / self.rho_dep)
            
            # # liftoff condition
            self.check_liftoff()
            
 
            # # Movie array update 
            if t >= printat:
                print(t, courant, Simulation['dt'])
                printat += self.tsim / self.numframes
                self.update_output_movies()
                movie_idx += 1

            # # Clear all variables
            [betax, betay, ulx, urx, vlx, vrx, uly, ury, vly, vry, S_x, S_y, Smaxtemp, tfluxx, tfluxy] = self.clearVariables([betax, betay, ulx, urx, vlx, vrx, uly, ury, vly, vry, S_x, S_y, Smaxtemp, tfluxx, tfluxy])
            
            # # Write output files
            self.create_output_files()