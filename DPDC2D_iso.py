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
                      centerYgeo = str(volc_params['centerYgeo']),
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
        self.ny = grid.ny
        self.dx = grid.dx
        self.dy = grid.dy
        self.topo = grid.elev
        
        self.centerXgeo = np.array(inparams['centerXgeo'].split(','), dtype=np.float32)
        self.centerYgeo = np.array(inparams['centerYgeo'].split(','), dtype=np.float32)
        self.centerX, self.centerY, self.numcells = self.set_domain()
        
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
        self.gzx = np.zeros((self.ny, self.nx), dtype = np.float64)
        self.gzy = np.zeros((self.ny, self.nx), dtype = np.float64)
        
        self.maxvel = np.zeros((self.ny, self.nx), dtype = np.float64)
        self.maxdepth = np.zeros((self.ny, self.nx), dtype = np.float64)
                
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
    
    def set_domain(self):
        
        px = np.zeros((1, len(self.centerXgeo)), dtype=int)
        py = np.zeros((1, len(self.centerYgeo)), dtype=int)
        
        for i in range(len(self.centerXgeo)):
            if self.grid.transform != None:
                px[0,i] = int((self.centerXgeo[i] - self.grid.transform[2])//self.dx)
                py[0,i] = int((self.centerYgeo[i] - self.grid.transform[5])//self.dy)
            else:
                px[0,i] = int(self.centerXgeo[i])
                py[0,i] = int(self.centerYgeo[i])
        coords = np.transpose(np.concatenate((px, py), axis=0))
        coords_uniq = np.unique(coords, axis = 0)
        return coords_uniq[:,0], coords_uniq[:,1], coords_uniq.shape[0]
        
    
    def set_initial_cond(self):
                
        ### Intitalizing conservative variables ######################
        
        # # Recalculating air density based on pre layer thickness h_film
        self.rho_a = self.hfilm * self.rho_s + (1-self.hfilm) * self.rho_a
        self.rho += self.rho_a

        for i in range(len(self.centerX)):
            self.h[self.centerY[i], self.centerX[i]] = self.hinit
#             self.h_dep = self.h
            self.rho[self.centerY[i], self.centerX[i]] =  self.phi_s0*self.rho_s + (1-self.phi_s0)*self.rho_g
        
            self.phi_s[self.centerY[i], self.centerX[i]] = self.phi_s0
            self.phi_s[self.centerY[i], self.centerX[i]] = self.phi_s0
            self.mass[self.centerY[i], self.centerX[i]] = self.rho[self.centerY[i], self.centerX[i]] * self.hinit
            
            self.velx[self.centerY[i], self.centerX[i]] = self.velocity * np.cos(self.direction)
            self.vely[self.centerY[i], self.centerX[i]] = self.velocity * np.sin(self.direction)
            self.vel[self.centerY[i], self.centerX[i]] = self.velocity


        volume = len(self.centerX) * np.abs(self.dx*self.dy)*self.hinit
        with open('/home/indujaa/pysurge/volume.txt', 'w') as f:
            f.write(str(volume))
            
        self.momx = self.mass * self.velx
        self.momy = self.mass * self.vely
        
        self.topo, self.h, self.h_dep, self.mass, self.momx, self.momy, self.rho, self.phi_s, self.velx, self.vely, self.vel, self.maxvel, self.maxdepth = \
        self.make_ghostcells([self.topo, self.h, self.h_dep, self.mass, self.momx, self.momy, self.rho, self.phi_s, self.velx, self.vely, self.vel, self.maxvel, self.maxdepth], 0)
            
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
        self.slope = np.sqrt(self.slopex **2 + self.slopey **2 + 0.001)                # True slope. Adding 1 to avoid division by zero
        
        self.gy = self.g * self.slopey / self.slope                   # Gravitational forcing in y direction
#         sy = - gy * u[0,:,:]                                        # Sorce term for momentum equation in y
        self.gx = self.g * self.slopex / self.slope                   # Gravitational forcing in x direction
    
        # # mew gravitational forcing according to Kelfoun (2007)
        self.gx = self.g * np.sin(np.arctan(self.slopex))
        self.gy = self.g * np.sin(np.arctan(self.slopey))
        
        self.gzx = self.g * np.cos(np.arctan(self.slopex))
        self.gzy = self.g * np.cos(np.arctan(self.slopey))

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
        self.gzx[:,0] = self.gzx[:,1]
        self.gzy[:,0] = self.gzy[:,1]
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
        self.gzx[:,-1] = self.gzx[:,-2]
        self.gzy[:,-1] = self.gzy[:,-2]
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
        self.gzx[0,:] = self.gzx[1,:]
        self.gzy[0,:] = self.gzy[1,:]
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
        self.gzx[-1,:] = self.gzx[-2,:]
        self.gzy[-1,:] = self.gzy[-2,:]
        self.velx[-1,:] = self.velx[-2,:] 
        self.u[:,-1,:] = self.u[:,-2,:]
        if LowerSolid:
            self.vely[-1,:] = - self.vely[-2,:]
            self.u[3,-1,:] = - self.u[3,-2,:]
        else:
            self.vely[-1,:] = self.vely[-2,:]
            
            
    def compute_fluxes(self,direction, ul, ur, vl, vr, S_xy, g_z, Smaxtemp, tflux):
    
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
            mx = self.nx-1
            my = self.ny
        elif direction == 'y':
            bvhl = ulr [0,2,:,:].copy() 
            bvhr = ulr [1,2,:,:].copy()
            buhl = ulr [0,3,:,:].copy()
            buhr = ulr [1,3,:,:].copy()
            mx = self.nx
            my = self.ny-1


        hstar = np.zeros((my,mx), dtype = np.float64)
        ustar = np.zeros((my,mx), dtype = np.float64)

        Sl = S_temp[0,:,:]
        Sm = S_temp[1,:,:]
        Sr = S_temp[2,:,:]


        hl_flow = hl > self.hfilm
        hl_noflow = hl <= self.hfilm

        hr_flow = hr > self.hfilm
        hr_noflow = hr <= self.hfilm

        ul[hl_flow] = buhl[hl_flow] / bhl[hl_flow]
        vl[hl_flow] = bvhl[hl_flow] / bhl[hl_flow]

        ur[hr_flow] = buhr[hr_flow] / bhr[hr_flow]
        vr[hr_flow] = bvhr[hr_flow] / bhr[hr_flow]


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
        
        tflux[0,:,:][cond_l] = hl[cond_l] * ul[cond_l]
        tflux[1,:,:][cond_l] = bhl[cond_l] * ul[cond_l]
        tflux[2,:,:][cond_l] = tflux_buhl
        
        tflux[0,:,:][cond_m] = (Sr[cond_m]*(hl[cond_m]*ul[cond_m]) - Sl[cond_m]*(hr[cond_m]*ur[cond_m]) + Sl[cond_m]*Sr[cond_m]*(hr[cond_m]-hl[cond_m])) / (Sr[cond_m]-Sl[cond_m])
        tflux[1,:,:][cond_m] = (Sr[cond_m]*(bhl[cond_m]*ul[cond_m]) - Sl[cond_m]*(bhr[cond_m]*ur[cond_m]) + Sl[cond_m]*Sr[cond_m]*(bhr[cond_m]-bhl[cond_m])) / (Sr[cond_m]-Sl[cond_m])
        tflux[2,:,:][cond_m] = (Sr[cond_m]*(hl[cond_m] * -self.rho_a * g_z[cond_m] * hl[cond_m]) + (bhl[cond_m] * (-ul[cond_m] ** 2 + 0.5 * g_z[cond_m] * hl[cond_m])) + (bhl[cond_m]  * 2*ul[cond_m]**2) - \
                                Sl[cond_m]*(hr[cond_m] * -self.rho_a * g_z[cond_m] * hr[cond_m]) + (bhr[cond_m] * (-ur[cond_m] ** 2 + 0.5 * g_z[cond_m] * hr[cond_m])) + (bhr[cond_m]  * 2*ur[cond_m]**2) \
                                + Sl[cond_m]*Sr[cond_m]*(bhr[cond_m]*ur[cond_m] - bhl[cond_m]*ul[cond_m])) / (Sr[cond_m]-Sl[cond_m])

        tflux[0,:,:][cond_r] = hr[cond_r] * ur[cond_r]
        tflux[1,:,:][cond_r] = bhr[cond_r] * ur[cond_r]
        tflux[2,:,:][cond_r] = tflux_buhr

        cond2_gt = Sm >= 0
        cond2_lt = Sm < 0
        tflux[3,:,:][cond2_gt] =  tflux[1,:,:][cond2_gt] * vl[cond2_gt]
        tflux[3,:,:][cond2_lt] = tflux[1,:,:][cond2_lt] * vr[cond2_lt]


        ### Maximum wave speeds for stability criteria #####
#         Smaxtemp = np.maximum(np.absolute(Sl),np.absolute(Sr), np.absolute(Sm))                             # # gives nan values
        Smaxtemp = max(np.nanmax(np.absolute(Sl)),np.nanmax(np.absolute(Sr)), np.nanmax(np.absolute(Sm)) )    # # getting rid of non values (ADDRESS: why nan; is this right)
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
    
    def gravity_term(self, dir):

        if dir == 'x':
            return (self.rho - self.rho_a)/self.rho_a * self.gx * self.u[0,:,:]
        elif dir == 'y':
            return (self.rho - self.rho_a)/self.rho_a * self.gy * self.u[0,:,:]
        
    def settling_term(self, cond):
        
        ws = np.zeros_like(self.rho)
        
#         print("density check = ", np.any((self.rho_s - self.rho[cond])<0))
        ws[cond] = np.sqrt((4 * (self.rho_s - self.rho[cond]) * self.g * self.d50) / (3 * self.Cdrag * self.rho[cond]))
#         print("settl vel =", np.nanmean(ws[cond]))

        # # add velocity while using for the momentum equations
        
        # # ADDRESS: should this be solid density or current density
#         return ws * self.rho_s * self.phi_s
        return  ws * (self.rho - self.rho_g)

        
    def drag_term(self, dir):
        dtx = np.zeros_like(self.velx) 
        dty = np.zeros_like(self.vely) 
        if dir == 'x':
#             dtx[self.velx > 0] = 1 * self.fric * self.rho[self.velx > 0] * self.velx[self.velx > 0] ** 2
#             dtx[self.velx < 0] = -1 * self.fric * self.rho[self.velx < 0] * self.velx[self.velx < 0] ** 2
            
            dtx = self.fric * self.rho * self.velx * np.abs(self.velx)
            return dtx
        elif dir == 'y':
#             dty[self.vely > 0] = 1 * self.fric * self.rho[self.vely > 0] * self.vely[self.vely > 0] ** 2
#             dty[self.vely < 0] = -1 * self.fric * self.rho[self.vely < 0] * self.vely[self.vely < 0] ** 2
            
            dty = self.fric * self.rho * self.vely * np.abs(self.vely)
            return dty
                    
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
        self.movie_velx = np.zeros((self.numframes, self.ny, self.nx), dtype=np.float64)
        self.movie_vely = np.zeros((self.numframes, self.ny, self.nx), dtype=np.float64)
        self.movie_h_dep = np.zeros((self.numframes, self.ny, self.nx), dtype=np.float64)
        self.movie_rho = np.zeros((self.numframes, self.ny, self.nx), dtype=np.float64)
        
    def update_output_movies(self, idx):
        print(idx)
        self.movie_h[idx,:,:] = self.u[0,1:-1,1:-1]
        self.movie_momx[idx,:,:] = self.u[2,1:-1,1:-1]
        self.movie_momy[idx,:,:] = self.u[3,1:-1,1:-1]
        self.movie_mass[idx,:,:] = self.u[1,1:-1,1:-1]
        self.movie_vel[idx,:,:] = self.vel[1:-1,1:-1]
        self.movie_velx[idx,:,:] = self.velx[1:-1,1:-1]
        self.movie_vely[idx,:,:] = self.vely[1:-1,1:-1]
        self.movie_h_dep[idx,:,:] = self.h_dep[1:-1,1:-1]
        self.movie_rho[idx,:,:] = self.rho[1:-1,1:-1]
    
    def output_variables(self):
                
#         cond = self.u[0,:,:]>self.hfilm
#         no_cond = self.u[0,:,:]<=self.hfilm    
 
        # # height and mass terms
#         self.u[0,:,:][no_cond] = self.hfilm
#         self.u[1,:,:][no_cond] = self.rho_a * self.hfilm
        self.rho = self.u[1,:,:] / self.u[0,:,:]
        
        # # liftoff condition based on density and surge height
#         surge = (self.u[0,:,:]>self.hfilm) & (self.rho > self.rho_a)
        surge = self.rho > self.rho_a
        surge_liftoff = np.logical_not(surge)
        
        self.phi_s[surge] = (self.rho[surge]-self.rho_g) / (self.rho_s - self.rho_g)
        self.phi_s[surge_liftoff] = self.hfilm
        
        
        self.u[0,:,:][surge_liftoff] = self.hfilm
        self.u[1,:,:][surge_liftoff] = self.rho_a * self.hfilm       
        self.rho = self.u[1,:,:] / self.u[0,:,:]

        # # momentum terms
        self.u[2,:,:][surge_liftoff] = 0    
        self.u[3,:,:][surge_liftoff] = 0  
        # # velocity
        self.velx = self.u[2,:,:] / self.u[1,:,:]
        self.vely = self.u[3,:,:]  / self.u[1,:,:]
        self.velx[surge_liftoff] = 0
        self.vely[surge_liftoff] = 0
        self.vel = self.vel_total()
        

        print("height = ", np.max(self.u[0,:,:][surge]))
#         print("mass = ", np.mean(self.u[1,:,:][surge]), "    ", np.any(self.u[0,:,:][surge]<0), "     ", self.u[1,:,:][31,12])
#         print("density = ", np.mean(self.rho[surge]), "    ", np.any(self.rho[surge]<0), "     ", self.rho[31,12])
#         print("velocity = ", np.mean(self.vel[surge]), "    ", np.any(self.vel[surge]<0), "     ", self.u[2,:,:][31,12])

        self.maxvel = np.maximum(self.maxvel, self.vel)
        self.maxdepth = np.maximum(self.maxdepth, self.u[0,:,:])


    def check_liftoff(self, t):
        # # ADDRESS: Does it need to be changed to account for only flow front density reversal?
        
        current_cond = self.u[0,:,:]>self.hfilm
        liftoff_cond = (self.u[0,:,:]>self.hfilm) & (self.rho <= self.rho_a)
        
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
        self.grid.write_tiff(self.u[0,1:-1,1:-1], self.outfiles_dir+"Depth.tif", -9999, 1, self.hfilm)
        self.grid.write_tiff(self.u[1,1:-1,1:-1], self.outfiles_dir+"Mass.tif", -9999, 1, self.rho_a)
        self.grid.write_tiff(self.u[2,1:-1,1:-1], self.outfiles_dir+"MomentumX.tif", -9999, 1, 0)
        self.grid.write_tiff(self.u[3,1:-1,1:-1], self.outfiles_dir+"MomentumY.tif", -9999, 1, 0)
        self.grid.write_tiff(self.h_dep[1:-1,1:-1], self.outfiles_dir+"Deposit_depth.tif", -9999, 1, self.hfilm)
        self.grid.write_tiff(self.rho[1:-1,1:-1], self.outfiles_dir+"Density.tif", -9999, 1, self.rho_a)
        self.grid.write_tiff(self.vel[1:-1,1:-1], self.outfiles_dir+"Velocity.tif", 0, 1, .001)
        self.grid.write_tiff(self.maxdepth[1:-1,1:-1], self.outfiles_dir+"MaxDepth.tif", -9999, 1, self.hfilm)
        self.grid.write_tiff(self.maxvel[1:-1,1:-1], self.outfiles_dir+"MaxVelocity.tif", 0, 1)


        #### Final results as movie files #########
        self.grid.write_tiff(self.movie_h, self.outfiles_dir+"DepthMovie.tif", -9999, self.numframes, self.hfilm)
#         self.grid.write_tiff(self.movie_momx, self.outfiles_dir+"MomentumXMovie.tif", 0, self.numframes, 0)
#         self.grid.write_tiff(self.movie_momy, self.outfiles_dir+"MomentumYMovie.tif", 0, self.numframes, 0)
        self.grid.write_tiff(self.movie_velx, self.outfiles_dir+"VelocityXMovie.tif", 0, self.numframes, .001)
        self.grid.write_tiff(self.movie_vely, self.outfiles_dir+"VelocityYMovie.tif", 0, self.numframes, .001)
        self.grid.write_tiff(self.movie_vel, self.outfiles_dir+"VelocityMovie.tif", 0, self.numframes, .001)
        self.grid.write_tiff(self.movie_h_dep, self.outfiles_dir+"DepositMovie.tif", -9999, self.numframes, self.hfilm)
        self.grid.write_tiff(self.movie_rho, self.outfiles_dir+"DensityMovie.tif", -9999, self.numframes, self.hfilm)
        
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
        
#         nsim = 0
#         while nsim<7:
        
        while t < self.tsim:
            self.update_boundary_cond()
            print("time = ", t)

            Smax = 0

            # # Flux in X-direction 
            [Smaxtemp, tfluxx, hstarx, ustarx] = self.compute_fluxes('x', ulx, urx, vlx, vrx, S_x, self.gzx[:,:-1], Smaxtemp, tfluxx)
            Smax = max(Smax, np.max(Smaxtemp))
            
            # # Flux in Y-direction 
            [Smaxtemp, tfluxy, hstary, ustary] = self.compute_fluxes('y', vly, vry, uly, ury, S_y, self.gzy[:-1,:], Smaxtemp, tfluxy)
            Smax = max(Smax, np.max(Smaxtemp))


            # # Adaptive time step 
            courant = self.adaptive_timestep(t, Smax)   

            if courant > 1:
                print("CFL error at t = ", t)
                t = 2 * self.tsim
            t += self.dt
            
            # # Source terms for momentum
            cond = self.u[0,:,:]>self.hfilm
            ws = self.settling_term(cond)
            
            betax = self.gravity_term('x') - ws*self.velx - self.drag_term('x')
            betay = self.gravity_term('y') - ws*self.vely - self.drag_term('y')
            
            # # Time update
#             self.u[0,1:-1,1:-1] = self.update_CV(self.u[0,1:-1,1:-1], tfluxx[0,1:-1,:], tfluxy[0,:,1:-1])
#             self.u[1,1:-1,1:-1] = self.update_CV(self.u[1,1:-1,1:-1], tfluxx[1,1:-1,:], tfluxy[1,:,1:-1], -1*ws[1:-1,1:-1])
#             self.u[2,1:-1,1:-1] = self.update_CV(self.u[2,1:-1,1:-1], tfluxx[2,1:-1,:], tfluxy[3,:,1:-1], betax[1:-1,1:-1])
#             self.u[3,1:-1,1:-1] = self.update_CV(self.u[3,1:-1,1:-1], tfluxx[3,1:-1,:], tfluxy[2,:,1:-1], betay[1:-1,1:-1])
            
            
            # # # # debugging # # # # ##############################
            GX =  self.gravity_term('x')
            SX = ws
            DX = self.drag_term('x')
            oneoverdx = 1/self.dx
            self.u[0,1:-1,1:-1] = self.u[0,1:-1,1:-1]  - self.dt * oneoverdx * np.diff(tfluxx[0,1:-1,:], axis=-1) - self.dt * oneoverdx * np.diff(tfluxy[0,:,1:-1], axis=-2) 
            print(np.nanmax(self.u[0,1:-1,1:-1]), np.nanmax(self.dt * oneoverdx * np.diff(tfluxx[0,1:-1,:], axis=-1)), np.nanmax(self.dt * oneoverdx * np.diff(tfluxy[0,:,1:-1], axis=-2)))
            self.u[1,1:-1,1:-1] = self.u[1,1:-1,1:-1] - self.dt * oneoverdx * np.diff(tfluxx[1,1:-1,:], axis=-1) - self.dt * oneoverdx * np.diff(tfluxy[1,:,1:-1], axis=-2) + self.dt * (-1*ws[1:-1,1:-1])

            self.u[2,1:-1,1:-1] = self.u[2,1:-1,1:-1] - self.dt * oneoverdx * np.diff(tfluxx[2,1:-1,:], axis=-1) - self.dt * oneoverdx * np.diff(tfluxy[3,:,1:-1], axis=-2) + self.dt * betax[1:-1,1:-1]
    
            self.u[3,1:-1,1:-1] = self.u[3,1:-1,1:-1] - self.dt * oneoverdx * np.diff(tfluxx[3,1:-1,:], axis=-1) - self.dt * oneoverdx * np.diff(tfluxy[2,:,1:-1], axis=-2) + self.dt * betay[1:-1,1:-1]
        
#             print("velx RHS terms: ", "past vel: ", np.nanmax(self.u[2,1:-1,1:-1]), "xflux: ", np.nanmax(oneoverdx * np.diff(tfluxx[2,1:-1,:], axis=-1)), "yflux ", np.nanmax(oneoverdx * np.diff(tfluxy[3,:,1:-1], axis=-2)), "source ", np.nanmax(betax[1:-1,1:-1]))           
            bx = betax[1:-1,1:-1]
#             print("source term = ", np.nanmean(bx), "gravity term = ", np.nanmean(GX), "settling term = ", np.nanmean(SX), "drag term = ", np.nanmean(DX))
            
#             print("vely RHS terms: ", "past vel: ", np.nanmax(self.u[3,1:-1,1:-1]), "xflux: ", np.nanmax(oneoverdx * np.diff(tfluxy[2,:,1:-1], axis=-2)), "yflux ", np.nanmax(oneoverdx * np.diff(tfluxx[3,1:-1,:], axis=-1)), "source ", np.nanmax(betay[1:-1,1:-1]))
            by = betay[1:-1,1:-1]
#             print("source term = ", np.nanmean(by))
            #########################################################
            
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
            [betax, betay, ulx, urx, vlx, vrx, uly, ury, vly, vry, S_x, S_y, Smaxtemp, tfluxx, tfluxy] = self.clearVariables([betax, betay, ulx, urx, vlx, vrx, uly, ury, vly, vry, S_x, S_y, Smaxtemp, tfluxx, tfluxy])
            
        # # Write output files
        self.create_output_files()