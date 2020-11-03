#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 17:12:38 2020

@author: MarkusArbeit
"""

from __future__ import division, print_function
#from builtins import range
import unittest
import numpy as np

from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation, calculate_phi_s
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import bigaussian
from blond.beam.profile import Profile, CutOptions
from blond.llrf.beam_feedback import BeamFeedback
from blond.trackers.utilities import separatrix


    
    
# Run before every test
negativeEta = False
acceleration = True
singleRF = True

# Defining parameters -------------------------------------------------
# Bunch parameters
N_b = 1.e9           # Intensity
N_p = 100000         # Macro-particles
tau_0 = 50.e-9          # Initial bunch length, 4 sigma [s]

# Machine parameters
C = 1567.5           # Machine circumference [m]
p_1i = 3.e9         # Synchronous momentum [eV/c]
p_1f = 30.0e9      # Synchronous momentum, final
p_2f = 40.e9         # Synchronous momentum [eV/c]
p_2i = 60.e9      # Synchronous momentum, final
gamma_t = 31.6       # Transition gamma
alpha_1 = -1./gamma_t/gamma_t  # First order mom. comp. factor
alpha_2 = 1./gamma_t/gamma_t  # First order mom. comp. factor

# RF parameters
h = [9,18]         # Harmonic number
V = [1800.e3,110.e3] # RF voltage [V]
phi_1 = [np.pi+1.,np.pi/6+2.]  # Phase modulation/offset
phi_2 = [1.,np.pi/6+2.]        # Phase modulation/offset
N_t = 43857

# Defining classes ----------------------------------------------------
# Define general parameters
if( negativeEta == True ): 

    if acceleration == True:
        # eta < 0, acceleration
        general_params = Ring(C, alpha_1, 
            np.linspace(p_1i, p_1f, N_t + 1), Proton(), N_t)
    elif acceleration == False: 
        # eta < 0, deceleration
        general_params = Ring(C, alpha_1, 
            np.linspace(p_1f, p_1i, N_t + 1), Proton(), N_t)

    if singleRF == True:
        rf_params = RFStation(general_params, 9, 1.8e6, np.pi+1.,
                              n_rf=1)
    elif singleRF == False:
        rf_params = RFStation(general_params, h, V, phi_1, n_rf=2)
        rf_params.phi_s = calculate_phi_s(
            rf_params, Particle=general_params.Particle,
            accelerating_systems='all')

elif( negativeEta == False ): 

    if acceleration == True:
        # eta > 0, acceleration
        general_params = Ring(C, alpha_2, 
            np.linspace(p_2i, p_2f, N_t + 1), Proton(), N_t)
    elif acceleration == False: 
        # eta > 0, deceleration
        general_params = Ring(C, alpha_2, 
            np.linspace(p_2f, p_2i, N_t + 1), Proton(), N_t)
        
    if singleRF == True:
        rf_params = RFStation(general_params, 9, 1.8e6, 1., n_rf=1)
    elif singleRF == False:
        rf_params = RFStation(general_params, h, V, phi_2, n_rf=2)
        rf_params.phi_s = calculate_phi_s(
            rf_params, Particle=general_params.Particle,
            accelerating_systems='all')

# Define beam and distribution
beam = Beam(general_params, N_p, N_b)
bigaussian(general_params, rf_params, beam, tau_0/4, seed = 1234) 
#print(np.mean(beam.dt))
slices = Profile(beam, CutOptions(cut_left=0.e-9, cut_right=600.e-9, 
                                  n_slices=1000))
slices.track()
configuration = {'machine': 'LHC', 
                 'PL_gain': 0.1*general_params.t_rev[0]}
PL = BeamFeedback(general_params, rf_params, slices, configuration)
PL.beam_phase()

# Quantities to be compared
phi_s = rf_params.phi_s[0]
phi_b = PL.phi_beam
phi_rf = rf_params.phi_rf[0,0]
dE_sep = separatrix(general_params, rf_params, [-5.e-7,-3.e-7,1.e-7,3.e-7,7.e-7,9.e-7])



        # self.setUp(negativeEta = False, acceleration = True, singleRF = True)

print(f"{phi_s:1.4f}, 3.3977")
print(f"{phi_b:1.3f}, 3.3978")
        # self.assertAlmostEqual(self.phi_s, 3.3977, places = 3, 
        #     msg = 'Failed test_5 in TestSeparatrixBigaussian on phi_s')
        # self.assertAlmostEqual(self.phi_b, 3.3978, places  = 3,
        #     msg = 'Failed test_5 in TestSeparatrixBigaussian on phi_b')
        # self.assertAlmostEqual(self.phi_rf, 1.0000, places = 3,
        #     msg = 'Failed test_5 in TestSeparatrixBigaussian on phi_rf')
        # self.assertAlmostEqual(self.dE_sep[0], 1.04542867e+09, delta = 1.e3,
        #     msg = 'Failed test_5 in TestSeparatrixBigaussian on dE_sep[0]')
        # self.assertAlmostEqual(self.dE_sep[1], 2.34232667e+09, delta = 1.e3,
        #     msg = 'Failed test_5 in TestSeparatrixBigaussian on dE_sep[1]')
        # self.assertAlmostEqual(self.dE_sep[2], 1.51776652e+09, delta = 1.e3,
        #     msg = 'Failed test_5 in TestSeparatrixBigaussian on dE_sep[2]')
        # self.assertAlmostEqual(self.dE_sep[3], 2.20889395e+09, delta = 1.e3,
        #     msg = 'Failed test_5 in TestSeparatrixBigaussian on dE_sep[3]')
        # self.assertAlmostEqual(self.dE_sep[4], 1.84686032e+09, delta = 1.e3,
        #     msg = 'Failed test_5 in TestSeparatrixBigaussian on dE_sep[4]')
        # self.assertAlmostEqual(self.dE_sep[5], 2.04363122e+09, delta = 1.e3,
        #     msg = 'Failed test_5 in TestSeparatrixBigaussian on dE_sep[5]')

        

