###################################################################
#
#                     LEGEND IQN PSD Emulator
#       github:
#       Intelectual property of:
#
#       utils.py ---> set of functions for PSD emulator
#
###################################################################


import uproot
import pandas as pd
import numpy as np
import json
import awkward as ak
import os
from math import floor
import h5py
from tqdm import tqdm


#### Plot formatting ####
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec


### Define some LEGEND specific variables
#GED_MAGE_ID = 1010503
#GED_NAME = "V09372A"
#GED_HEIGHT = 111.8
GED_MAGE_ID = 1010908
GED_NAME = "P00661C"
GED_HEIGHT = 42.1


def gaus(x, mu, sig, A, h):
    return 1/np.sqrt(2*np.pi*sig**2)*np.exp(-0.5*((x-mu)/sig)**2)*A + h


def linear(x, m, c):
    return m*x + c


def create_PET_file(pet_filename):
    if len(pet_filename.split('.csv')) == 1:
        msg = '> PET file must be a .csv file format'
        raise IOError(msg)
    with open(pet_filename, 'w') as outfile:
        outfile.write(
            "#class tools::wcsv::ntuple\n"
            + "#title Energy and time\n"
            + "#separator 44\n"
            + "#vector_separator 59\n"
            + "#column double Event#\n"
            + "#column double Detector_ID\n"
            + "#column double Time\n"
            + "#column double EnergyDeposit\n"
            + "#column double X\n"
            + "#column double Y\n"
            + "#column double Z\n"
        )


def append_PET_file(pet_filename, dict_data):
    with open(pet_filename, 'a') as outfile:
        xs = dict_data['x'][-1]
        ys = dict_data['y'][-1]
        zs = dict_data['z'][-1]
        rs = dict_data['r'][-1]
        ts = dict_data['t'][-1]
        Es = dict_data['E'][-1]
        ns = dict_data['ievt_n'][-1]
        dts = dict_data['dts'][-1]
        sumE = dict_data['Esum'][-1]
        
        if isinstance(Es, list):
            for index in range(len(Es)):
                outfile.write(
                    f"{ns[index]}, 1, {ts[index]}, {Es[index]}, {xs[index]}, {ys[index]}, {zs[index]}\n"
                )
        else:
            outfile.write(
                f"{ns}, 1, {ts}, {Es}, {xs}, {ys}, {zs}\n"
            )


def nested_drift_time(fn, r_array, z_array):
    if isinstance(r_array, ak.Array) and isinstance(z_array, ak.Array):
        return ak.Array([nested_drift_time(fn, r, z) for r, z in zip(r_array, z_array)])
    else:
        return fn(r_array, z_array)


class DriftTimes:
    def __init__(self, r, z, t) -> None:
       self.r = r
       self.z = z
       self.t = t


class ReadDTFile:
    def __init__(self, filename) -> None:
        datatype = filename.split('/')[-1].split(".")[-1]
        self.r = []
        self.z = []
        self.t = []
        self.drift_times = []
        self.filename = filename

        with open(filename, "r") as file:
            fl = file.readlines()

            for index, line in enumerate(fl):
                if datatype == 'dat':
                    line_list = line.split(" ")
                else:
                    line_list = line.split(",")
                if index == 0: continue

                r = float(line_list[0])*1000
                z = float(line_list[1])*1000
                t = float(line_list[2].strip())

                self.r.append(r)
                self.z.append(z)
                self.t.append(t)

                self.drift_times.append(DriftTimes(r, z, t))

        rs, zs = [], []
        for xi, zi in zip(self.r,self.z):
            if xi not in rs: rs.append(xi)
            if zi not in zs: zs.append(zi)
        self.dz = abs(zs[-1] - zs[-2])
        self.dr = abs(rs[-1] - rs[-2])
        self.maxz = max(zs)
        self.maxr = max(rs)
        self.minz = min(zs)
        self.minr = min(rs)

    def GetIndex(self, r, z):
        rGrid = floor(r/self.dr+0.5)*self.dr;
        zGrid = floor(z/self.dz+0.5)*self.dz;
        numZ = int((self.maxz-self.minz)/self.dz+1)
        i = int((rGrid-self.minr)/self.dr*numZ + (zGrid-self.minz)/self.dz)
        nDT = len(self.drift_times)

        if i>=nDT:
            return nDT-1
        return i

    def GetTime(self, r, z):
        
        r1 = floor(r/self.dr)*self.dr;
        z1 = floor(z/self.dz)*self.dz;
  
        t11 = self.drift_times[self.GetIndex(r1, z1)].t
        if r1==r and z1==z: return t11

        # Bilinear interpolation. Might do cubic interpolation eventually.
        r2 = r1+self.dr
        z2 = z1+self.dz

        t12 = self.drift_times[self.GetIndex(r1, z2)].t
        t21 = self.drift_times[self.GetIndex(r2, z1)].t
        t22 = self.drift_times[self.GetIndex(r2, z2)].t

        return 1/((r2-r1)*(z2-z1))*(t11*(r2-r)*(z2-z)+t21*(r-r1)*(z2-z)+t12*(r2-r)*(z-z1)+t22*(r-r1)*(z-z1))

    def save_dat_file(self, is_mpp: bool):
        dat_filename = self.filename.split('/')[-1].split('.')[0]
        if is_mpp:
            dat_filename += '_mpp.dat'
        else:
            dat_filename += '.dat'
        
        with open(dat_filename, "w") as file:
            for r, z, t in zip(self.r, self.z, self.t):
                if is_mpp: z = z - 111.8/2
                file.write(
                    f"{r} {z} {t}\n"
                )