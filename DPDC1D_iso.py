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
       

        params = dict(outpath = output['out_path'],
                      outdir = output['out_dir'],
                      totaltime = np.int(sim_params['total_time']),
                      numvents = np.int(sim_params['num_vents']),
                      numframes = np.int(sim_params['num_frames']),
                      timestep = np.float(sim_params['timestep']),
                      cstable = np.float(sim_params['cstable']),
                      maxstep = np.float(sim_params['maxstep']),
                      minstep = np.float(sim_params['minstep']),
                      order = np.int(sim_params['order']),
                      gravity = np.float(phys_params['gravity']),
                      rho_air = np.float(atmo_params['rho_air']), 
                      phi_colair = np.float(atmo_params['phi_colair']), 
                      mu_air = np.float(atmo_params['mu_air']),
                      Cp_air = np.float(atmo_params['Cp_air']),
                      rho_gas = np.float(volc_params['rho_gas']), 
                      mu_gas = np.float(volc_params['mu_gas']),
                      Cp_gas = np.float(volc_params['Cp_gas']),
                      rho_dep = np.float(volc_params['rho_deposit']),
                      rho_s = np.float(volc_params['rho_solid']), 
                      n_s = np.float(volc_params['n_solid']),
                      C_s = np.float(volc_params['C_solid']),
                      d50 = np.float(volc_params['d50']),
                      Cd = np.float(volc_params['Cd']),
                      fric = np.float(volc_params['fric']),
                      height = np.float(volc_params['height']),
                      centerXgeo = str(volc_params['centerXgeo']),
                      velocity = np.float(volc_params['velocity']),
                      direction = np.deg2rad(np.float(volc_params['direction'])))
                      

        params['hfilm'] = 1e-5
        
        return params
    

class DiluteCurrentModel():
    
    def __init__(self, inparams, grid):
        # # Setting model parameters
        self.tsim = inparams['totaltime']
        self.dt = inparams['timestep']
        self.cstable = inparams['cstable']
        self.maxdt = inparams['maxstep']
        self.mindt = inparams['minstep']
        self.order = inparams['order']
        
        self.numframes = inparams['numframes']
        
        self.hfilm = inparams['hfilm']
        
        self.g = inparams['gravity']
        
        self.rho_a = inparams['rho_air']
        self.mu_a = inparams['mu_air']
        self.Cp_a = inparams['Cp_air']
        self.phi_colair = inparams['phi_colair']
        
        self.rho_s = inparams['rho_s']
        self.phi_s0 = inparams['n_s']
        self.C_s = inparams['C_s']
        
        self.rho_dep = inparams['rho_dep']
        
        self.rho_g = inparams['rho_gas']
        self.phi_g = 1 - self.phi_s0
        
        self.mu_g = inparams['mu_gas']
        self.Cp_g = inparams['Cp_gas']

        
        self.d50 = inparams['d50']
        self.Cdrag = inparams['Cd']
        self.fric = inparams['fric']
        
        self.hinit = inparams['height']
        self.velocity = inparams['velocity']
        self.direction = inparams['direction']
        self.grid = grid
        self.nx = grid.nx
        self.dx = grid.dx
        self.topo = grid.elev.reshape(self.nx)
        
        # # change this to float and use lat/lon to pixel conversion if using real topo
        self.centerX =np.array(inparams['centerXgeo'].split(','), dtype=int)
        self.numcells = len(self.centerX)
        
        # # Initializing fields
        # # Need to add stuff for thermal and density
        self.nn = 3
        self.h = self.hfilm + np.zeros(self.nx, dtype = np.float64)
        self.h_dep = self.hfilm + np.zeros(self.nx, dtype = np.float64)
        self.mass = np.zeros(self.nx, dtype = np.float64)
        self.vel = np.zeros(self.nx, dtype = np.float64)
        self.mom = np.zeros(self.nx, dtype = np.float64)
        self.rho = np.zeros(self.nx, dtype = np.float64)
        self.phi_s = np.zeros(self.nx, dtype = np.float64)
        
        # # Inititalizing secondary variables
        self.slope = np.zeros(self.nx, dtype = np.float64)
        self.gx = np.zeros(self.nx, dtype = np.float64)
        self.gz = np.zeros(self.nx, dtype = np.float64)
        
        self.maxvel = np.zeros(self.nx, dtype = np.float64)
        self.maxdepth = np.zeros(self.nx, dtype = np.float64)
                
        # # Output files
        self.outfiles_dir = self.set_output_directory(inparams['outpath'], inparams['outdir'])
        
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
                if arraynd.ndim == 1:
                    arraynd = np.pad(arraynd, (1,1), mode='constant', constant_values=(const,const))
                    arraylist.append(arraynd)
                elif arraynd.ndim == 2:
                    arraynd = np.pad(arraynd, ((0,0), (1,1)), mode='constant', constant_values=((const,const),(const,const)))
                    arraylist.append(arraynd)
            return arraylist 

        elif type(arrays) == np.ndarray:
            arrays =  np.pad(arraynd, (1,1), mode='constant', constant_values=(const,const))
            return arrays
        
        
    def n_a(self):
        return 1 - self.n_g - self.n_s

        
    
    def set_initial_cond(self):
                
        ### Intitalizing conservative variables ######################
        
        # # Recalculating air density based on pre layer thickness h_film
        self.rho_a = self.hfilm * self.rho_s + (1-self.hfilm) * self.rho_a
        self.rho += self.rho_a

        for i in range(len(self.centerX)):
            self.h[self.centerX[i]] = self.hinit
#             self.h_dep = self.h
            self.rho[self.centerX[i]] =  self.phi_s0*self.rho_s + (1-self.phi_s0)*self.rho_g
        
            self.phi_s[self.centerX[i]] = self.phi_s0
            self.phi_s[self.centerX[i]] = self.phi_s0
            self.mass[self.centerX[i]] = self.rho[self.centerX[i]] * self.hinit
            
            self.vel[self.centerX[i]] = self.velocity * np.cos(self.direction)

        volume = np.abs((len(self.centerX) * self.dx * self.hinit))
        with open('/home/indujaa/pysurge/volume.txt', 'w') as f:
            f.write(str(volume))
            
        self.mom = self.mass * self.vel
        
        self.topo, self.h, self.h_dep, self.mass, self.mom, self.rho, self.phi_s, self.vel, self.maxvel, self.maxdepth = \
        self.make_ghostcells([self.topo, self.h, self.h_dep, self.mass, self.mom, self.rho, self.phi_s, self.vel, self.maxvel, self.maxdepth], 0)
            
        self.u = np.array([self.h, self.mass, self.mom])


        #### Set boundary conditions for topography #############
        ### Left ###
        self.topo[0] = self.topo [1]

        ### Right ###
        self.topo[-1] = self.topo [-2]


        self.slope = np.gradient(self.topo, self.dx)                     # Slope in x direction 
    
        # # mew gravitational forcing according to Kelfoun (2007)
        self.gx = self.g * np.sin(np.arctan(self.slope))        
        self.gz = self.g * np.cos(np.arctan(self.slope))
 
        self.maxdepth = self.u[0,:]
        self.maxvel = self.vel  

    
    def update_boundary_cond(self):
        
        """ Updates BCs at every time step for topo, gx, gy, gz (transmissive / solid not required)
         Updates BCs for conserved quanitites and velocity based on whether the boundary is solid or transmissive """

        LeftSolid = False
        RightSolid = False

        ## Left boundary 
        self.gx[0] = self.gx[1]
        self.gz[0] = self.gz[1]                      
        self.u[:,0] = self.u[:,1]
        if LeftSolid:
            self.vel[0] = - self.vel[1] 
            self.u[2,0] = - self.u[2,1]
        else:
            self.vel[0] = self.vel[1] 


        ## Right boundary
        self.gx[-1] = self.gx[-2]
        self.gz[-1] = self.gz[-2]
        self.u[:,-1] = self.u[:,-2]
        if RightSolid:
            self.vel[-1] = - self.vel[-2] 
            self.u[2,-1] = - self.u[2,-2]
        else:
            self.vel[-1] = self.vel[-2]  

            
    def compute_fluxes(self,direction, ul, ur, S_xy, g_z, Smaxtemp, tflux):
    
        ulr = self.MUSCLextrap()                  # ulr shape is nn x 2
        [Smaxtemp, tflux, hstar, ustar] = self.hllc(ulr, ul, ur, S_xy, g_z, direction, Smaxtemp, tflux)
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

    def hllc(self, ulr, ul, ur, S_temp, g_z, direction, Smaxtemp, tflux):
        """ Uses HLLC solver to calculate the fluxes at every cell interface (computation is done across the whole grid and not cell by cell).
        Returns the maximum wave speed, temporary flux, hstar and ustar"""

        ## Beware ulr is a 4D array. axis 0, size 2 - conserved quantity at i-1/2 and i+12; axis 1, size 3 - conserved quantities h, hu and hv;
        ## axis 3, size nrows - rows; axis 4, size ncols - values in (row, col)

        n = self.nn
        hl = ulr [0,0,:].copy()
        hr = ulr [1,0,:].copy() 
        bhl = ulr [0,1,:].copy() 
        bhr = ulr [1,1,:].copy()
        buhl = ulr [0,2,:].copy() 
        buhr = ulr [1,2,:].copy()
        mx = self.nx-1

        hstar = np.zeros(mx, dtype = np.float64)
        ustar = np.zeros(mx, dtype = np.float64)

        Sl = S_temp[0,:]
        Sm = S_temp[1,:]
        Sr = S_temp[2,:]


        hl_flow = hl > self.hfilm
        hl_noflow = hl <= self.hfilm
        hr_flow = hr > self.hfilm
        hr_noflow = hr <= self.hfilm

        ul[hl_flow] = buhl[hl_flow] / bhl[hl_flow]
        ur[hr_flow] = buhr[hr_flow] / bhr[hr_flow]


        hstar = (ul + (2 * np.sqrt(g_z*hl)) - ur - (2*np.sqrt(g_z*hr))) ** 2 / (16 * g_z)
        ustar = (0.5 * (ul + ur)) + np.sqrt(g_z*hl) - np.sqrt(g_z*hr)


        #### wave speeds ###################
        Sl, Sm, Sr = self.wave_speeds(ul, hl, hl_noflow, ur, hr, hr_noflow, ustar, hstar, g_z, Sl, Sm, Sr)

        ### Fluxes #############################
        cond_l = Sl >= 0
        cond_m = (Sl < 0) & (Sr > 0)
        cond_r = Sr <= 0
        
        tflux_buhl = (hl[cond_l] * -self.rho_a * g_z[cond_l] * hl[cond_l]) + (bhl[cond_l] * (-ul[cond_l] ** 2 + 0.5 * g_z[cond_l] * hl[cond_l])) + (bhl[cond_l]  * 2*ul[cond_l]**2)
        tflux_buhr = (hr[cond_r] * -self.rho_a * g_z[cond_r] * hr[cond_r]) + (bhr[cond_r] * (-ur[cond_r] ** 2 + 0.5 * g_z[cond_r] * hr[cond_r])) + (bhr[cond_r]  * 2*ur[cond_r]**2)
        
        tflux[0,:][cond_l] = hl[cond_l] * ul[cond_l]
        tflux[1,:][cond_l] = bhl[cond_l] * ul[cond_l]
        tflux[2,:][cond_l] = tflux_buhl
        
        tflux[0,:][cond_m] = (Sr[cond_m]*(hl[cond_m]*ul[cond_m]) - Sl[cond_m]*(hr[cond_m]*ur[cond_m]) + Sl[cond_m]*Sr[cond_m]*(hr[cond_m]-hl[cond_m])) / (Sr[cond_m]-Sl[cond_m])
        tflux[1,:][cond_m] = (Sr[cond_m]*(bhl[cond_m]*ul[cond_m]) - Sl[cond_m]*(bhr[cond_m]*ur[cond_m]) + Sl[cond_m]*Sr[cond_m]*(bhr[cond_m]-bhl[cond_m])) / (Sr[cond_m]-Sl[cond_m])
        tflux[2,:][cond_m] = (Sr[cond_m]*(hl[cond_m] * -self.rho_a * g_z[cond_m] * hl[cond_m]) + (bhl[cond_m] * (-ul[cond_m] ** 2 + 0.5 * g_z[cond_m] * hl[cond_m])) + (bhl[cond_m]  * 2*ul[cond_m]**2) - \
                                Sl[cond_m]*(hr[cond_m] * -self.rho_a * g_z[cond_m] * hr[cond_m]) + (bhr[cond_m] * (-ur[cond_m] ** 2 + 0.5 * g_z[cond_m] * hr[cond_m])) + (bhr[cond_m]  * 2*ur[cond_m]**2) \
                                + Sl[cond_m]*Sr[cond_m]*(bhr[cond_m]*ur[cond_m] - bhl[cond_m]*ul[cond_m])) / (Sr[cond_m]-Sl[cond_m])

        tflux[0,:][cond_r] = hr[cond_r] * ur[cond_r]
        tflux[1,:][cond_r] = bhr[cond_r] * ur[cond_r]
        tflux[2,:][cond_r] = tflux_buhr


        ### Maximum wave speeds for stability criteria #####
#         Smaxtemp = np.maximum(np.absolute(Sl),np.absolute(Sr), np.absolute(Sm))                             # # gives nan values
        Smaxtemp = max(np.nanmax(np.absolute(Sl)),np.nanmax(np.absolute(Sr)), np.nanmax(np.absolute(Sm)) )    # # getting rid of non values (ADDRESS: why nan; is this right)
        return Smaxtemp, tflux, hstar, ustar

    def MUSCLextrap(self):

        """ Monotonic Upwind Scheme blah blah something. Returns value at i-1 and i+1 for 1st order (I think) """
        val = self.u[:,:-1]
        valplus = self.u[:,1:]

        return np.array([val, valplus])

    def adaptive_timestep(self, currtime, Smax):
        
        self.dt = self.cstable * self.dx / Smax
#         print("dt = ", self.dt)
#         print("Smax = ", Smax)

        if self.dt > self.maxdt:
            self.dt = self.maxdt
        if currtime+self.dt > self.tsim:
            self.dt = self.tsim - currtime
        courant = Smax * self.dt / self.dx

        return courant 
    
    def gravity_term(self):

        return (self.rho - self.rho_a)/self.rho_a * self.gx * self.u[0,:]

        
    def settling_term(self, cond):
        
        ws = np.zeros_like(self.rho)
        ws[cond] = np.sqrt((4 * (self.rho_s - self.rho[cond]) * self.g * self.d50) / (3 * self.Cdrag * self.rho[cond]))
        return  ws * (self.rho - self.rho_g)

        
    def drag_term(self):

        return self.fric * self.rho * self.vel * np.abs(self.vel)
                    
    def update_CV(self, CV, fluxx, sourceterm=0):
        oneoverdx = 1/self.dx
        CV_new = CV - self.dt * oneoverdx * np.diff(fluxx) + self.dt * sourceterm

        return CV_new
    
    def set_tempvariables(self):
        #### Variables for storing shit
    
        h = np.zeros(self.nx, dtype = np.float64)

        #### Sourrce terms
        betax = np.zeros(self.nx, dtype = np.float64)

        #### Left and right quantities
        ulx = np.zeros(self.nx-1, dtype = np.float64)
        urx = np.zeros(self.nx-1, dtype = np.float64)

        #### Wave speeds
        S_x = np.zeros((self.nn,self.nx-1), dtype = np.float64)
        Smaxtemp = np.zeros(self.nx, dtype = np.float64)

        #### Fluxes
        tflux = np.zeros((self.nn,self.nx-1), dtype = np.float64)


        return h, betax, ulx, urx, S_x, Smaxtemp, tflux
    
    def clearVariables(self, arr_list):
        new_arr_list = []
        for arr in arr_list:
            arr *= 0
            new_arr_list.append(arr)

        return new_arr_list

    
    def create_output_movies(self):
        self.movie_h = np.zeros((self.numframes, self.nx), dtype=np.float64)
        self.movie_momx = np.zeros((self.numframes, self.nx), dtype=np.float64)
        self.movie_mass = np.zeros((self.numframes, self.nx), dtype=np.float64)
        self.movie_vel = np.zeros((self.numframes, self.nx), dtype=np.float64)
        self.movie_h_dep = np.zeros((self.numframes, self.nx), dtype=np.float64)
        self.movie_rho = np.zeros((self.numframes, self.nx), dtype=np.float64)
        
    def update_output_movies(self, idx):
        print(idx)
        self.movie_h[idx,:] = self.u[0,1:-1]
        self.movie_momx[idx,:] = self.u[2,1:-1]
        self.movie_mass[idx,:] = self.u[1,1:-1]
        self.movie_vel[idx,:] = self.vel[1:-1]
        self.movie_h_dep[idx,:] = self.h_dep[1:-1]
        self.movie_rho[idx,:] = self.rho[1:-1]
    
    def output_variables(self):
                
#         cond = self.u[0,:,:]>self.hfilm
#         no_cond = self.u[0,:,:]<=self.hfilm    
 
        # # height and mass terms
#         self.u[0,:,:][no_cond] = self.hfilm
#         self.u[1,:,:][no_cond] = self.rho_a * self.hfilm
        self.rho = self.u[1,:] / self.u[0,:]
        
        # # liftoff condition based on density and surge height
#         surge = (self.u[0,:,:]>self.hfilm) & (self.rho > self.rho_a)
        surge = self.rho > self.rho_a
        surge_liftoff = np.logical_not(surge)
        
        self.phi_s[surge] = (self.rho[surge]-self.rho_g) / (self.rho_s - self.rho_g)
        self.phi_s[surge_liftoff] = self.hfilm
        
        
        self.u[0,:][surge_liftoff] = self.hfilm
        self.u[1,:][surge_liftoff] = self.rho_a * self.hfilm       
        self.rho = self.u[1,:] / self.u[0,:]

        # # momentum terms
        self.u[2,:][surge_liftoff] = 0    
        # # velocity
        self.vel = self.u[2,:] / self.u[1,:]
        self.vel[surge_liftoff] = 0

        self.maxvel = np.maximum(self.maxvel, self.vel)
        self.maxdepth = np.maximum(self.maxdepth, self.u[0,:])


    def check_liftoff(self, t):
        # # ADDRESS: Does it need to be changed to account for only flow front density reversal?
        
        current_cond = self.u[0,:]>self.hfilm
        liftoff_cond = (self.u[0,:]>self.hfilm) & (self.rho <= self.rho_a)
        
        # # 1) using mean
        diff = np.nanmean(self.rho[current_cond] - self.rho_a)
        if (diff <= 0.1 or np.isnan(diff)):
            print("liftoff condition reached")
            t = 2 * self.tsim
            
        # # 2) using np.all 
#         if np.all(self.rho[current_cond] - self.rho_a <= 0.1):
#             print("liftoff condition reached")
#             t = 2 * self.tsim
            
        # # 3) setting surge height to 0
#         self.u[0,:,:][liftoff_cond] = self.hfilm
#         self.u[1,:,:][liftoff_cond] = self.hfilm * self.rho_a
            
        return t
    
    def create_output_files(self):
        #### Final results as binary files #########
        self.grid.write_txt1D(self.u[0,1:-1], self.outfiles_dir+"Depth.txt", -9999, 1, self.hfilm)
        self.grid.write_txt1D(self.u[1,1:-1], self.outfiles_dir+"Mass.txt", -9999, 1, self.rho_a)
        self.grid.write_txt1D(self.u[2,1:-1], self.outfiles_dir+"MomentumX.txt", -9999, 1, 0)
        self.grid.write_txt1D(self.h_dep[1:-1], self.outfiles_dir+"Deposit_depth.txt", -9999, 1, self.hfilm)
        self.grid.write_txt1D(self.rho[1:-1], self.outfiles_dir+"Density.txt", -9999, 1, self.rho_a)
        self.grid.write_txt1D(self.vel[1:-1], self.outfiles_dir+"Velocity.txt", 0, 1, .001)
        self.grid.write_txt1D(self.maxdepth[1:-1], self.outfiles_dir+"MaxDepth.txt", -9999, 1, self.hfilm)
        self.grid.write_txt1D(self.maxvel[1:-1], self.outfiles_dir+"MaxVelocity.txt", 0, 1)


        #### Final results as movie files #########
        self.grid.write_txt1D(self.movie_h, self.outfiles_dir+"DepthMovie.txt", -9999, self.numframes, self.hfilm)
#         self.grid.write_tiff1D(self.movie_momx, self.outfiles_dir+"MomentumXMovie.tif", 0, self.numframes, 0)
#         self.grid.write_tiff1D(self.movie_momy, self.outfiles_dir+"MomentumYMovie.tif", 0, self.numframes, 0)
        self.grid.write_txt1D(self.movie_vel, self.outfiles_dir+"VelocityXMovie.txt", 0, self.numframes, .001)
        self.grid.write_txt1D(self.movie_h_dep, self.outfiles_dir+"DepositMovie.txt", -9999, self.numframes, self.hfilm)
        self.grid.write_txt1D(self.movie_rho, self.outfiles_dir+"DensityMovie.txt", -9999, self.numframes, self.hfilm)
        
        # # ADDRESS: REWRITE FOR DEPOSIT THICKNESS

#         ## TIFF file for classification
#         u_bin = u[0,1:-1,1:-1].astype(np.int8)
#         u_bin[u_bin >= 0.1] = 1                                     ## binary raster with depth>20cm = 1 and depth<20cm = 0
#         u_bin[u_bin < 0.1] = 0
#         grid.write_tiff1D(u_bin, outdir+"Depthbin.tif", None, 1)
        
    
    def run_model(self):
        ##### Set Initial Conditions ######
        self.set_initial_cond()


        ### Arrays to hold movie time series ############################
        movie_idx = 0    
        self.create_output_movies()
        printat = self.tsim / self.numframes 


        h, betax, ulx, urx, S_x, Smaxtemp, tflux = self.set_tempvariables()
        ulx, urx, tflux, betax = self.make_ghostcells([ulx, urx, tflux, betax], 0)
        
        # # ADDRESS: SOURCE TERMS BETAX AND BETAY ARE DIFFERENT; NEED TO BE ADDED DIFFERENTLY

        t = 0
        
#         nsim = 0
#         while nsim<7:
        
        while t < self.tsim:
            self.update_boundary_cond()
            print("time = ", t)

            Smax = 0

            # # Flux in X-direction 
            [Smaxtemp, tflux, hstarx, ustarx] = self.compute_fluxes('x', ulx, urx, S_x, self.gz[:-1], Smaxtemp, tflux)
            Smax = max(Smax, np.max(Smaxtemp))


            # # Adaptive time step 
            courant = self.adaptive_timestep(t, Smax)   

            if courant > 1:
                print("CFL error at t = ", t)
                t = 2 * self.tsim
            t += self.dt
            
            # # Source terms for momentum
            cond = self.u[0,:]>self.hfilm
            ws = self.settling_term(cond)
            
            betax = self.gravity_term() - ws*self.vel - self.drag_term()
            
            # # Time update
            self.u[0,1:-1] = self.update_CV(self.u[0,1:-1], tflux[0])
            self.u[1,1:-1] = self.update_CV(self.u[1,1:-1], tflux[1], -1*ws[1:-1])
            self.u[2,1:-1] = self.update_CV(self.u[2,1:-1], tflux[2], betax[1:-1])
            
            
            # # ADDRESS: referenced before assignment
            self.output_variables()
            
            # # deposit thickness
            # # ADDRESS: should the thickness be calculated based on density condition or hfilm condition
            current_cond = self.rho > self.rho_a
            self.h_dep[current_cond] = self.h_dep[current_cond] + (self.dt * self.settling_term(cond)[current_cond] / self.rho_dep)
            
            # # liftoff condition
            t = self.check_liftoff(t)
            
 
            # # Movie array update 
            if t >= printat:
                print("Time      Courant      Timestep")
                print(t, courant, self.dt)
                printat += self.tsim / self.numframes
                self.update_output_movies(movie_idx)
                movie_idx += 1
                
#             ### debugging ###
#             self.update_output_movies(nsim)
#             nsim += 1
#             #################

            # # Clear all variables
            [betax, ulx, urx, S_x, Smaxtemp, tflux] = self.clearVariables([betax, ulx, urx, S_x, Smaxtemp, tflux])
            
        # # Write output files
        self.create_output_files()