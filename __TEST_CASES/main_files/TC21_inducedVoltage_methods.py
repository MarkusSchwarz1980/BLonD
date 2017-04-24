# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3), 
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities 
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
Compares the different methods to compute the induced voltage of several resonators.
'''
# General Imports
import numpy as np
import matplotlib.pyplot as plt

# BLonD imports
from input_parameters.general_parameters import GeneralParameters
from input_parameters.rf_parameters import RFSectionParameters
from beams.beams import Beam
from beams.distributions import longitudinal_bigaussian
from trackers.tracker import RingAndRFSection
from trackers.tracker import FullRingAndRF
from beams.slices import Slices
from impedances.impedance_sources import Resonators
from impedances.impedance import InducedVoltageFreq
from impedances.impedance import InducedVoltageTime
from impedances.impedance import TotalInducedVoltage
from impedances.impedance import InducedVoltageResonator
from impedances.induced_voltage_analytical import analytical_gaussian_resonator

# Set up beam
particle_type = 'proton' #particle species
circumference = 6911.5038   #circumference of SPS [m]
p_init = 25.91e9 #initial momentum [eV/c]
gamma_t = 17.95  # Transition gamma of SPS from Q20
alpha = 1./gamma_t/gamma_t        # First order mom. comp. factor
n_turns=1 #we don't do any real tracking
general_params = GeneralParameters(n_turns, circumference, alpha,p_init,particle_type)

# Simple model for constant momentum
h = 4620 #harmonic number [1]
V = 1e6 #cavity voltage [V]
phi_offset = 0 #phase [rad]
num_rf = 1 #one rf station
rf_params = RFSectionParameters(general_params,num_rf,h,V,phi_offset)

n_macroparticles = 10000 #number of macroparticles [1]
beam_intensity = 1e11 #number of particles per bunch [1]
bunch_sigma = rf_params.t_RF[0]/8. #bunch length [s]

beam = Beam(general_params,n_macroparticles,beam_intensity)
# We are interested in checking against the analytical result of a Gaussian
# To get reproducible results, seed the number generator
longitudinal_bigaussian(general_params,rf_params,beam,bunch_sigma,seed=1980)
beam.statistics()

rf_sections = RingAndRFSection(rf_params,beam)
ring = FullRingAndRF([rf_sections])

# Compute the line density
n_sclices = 42
slices = Slices(rf_params, beam, \
                n_sclices,cut_left=0.0,cut_right=2*np.pi,cuts_unit='rad')
slices.track()

#We use two resonators; the resonant frequencies are chosen to have some overlap
#with the line density
Q = np.array([10,100])
omega_r = np.array([2/(bunch_sigma),1/(bunch_sigma)])
R = np.array([1,1])

#Create the Resonator object
imp_source = Resonators(R,omega_r/(2*np.pi),Q)

# First, calculate the induced voltage based on InducedVoltageFreq
tmpFreq = InducedVoltageFreq(beam,slices,np.array([imp_source]),
                          frequency_resolution=144e3)
tmpFreq.induced_voltage_1turn()
VindFreq = TotalInducedVoltage(beam,slices,np.array([tmpFreq]))
VindFreq.induced_voltage_sum()

# Second, calculate the induced voltage based on InducedVoltageTime
tmpTime = InducedVoltageTime(beam,slices,np.array([imp_source]))
tmpTime.induced_voltage_1turn()
VindTime = TotalInducedVoltage(beam,slices,np.array([tmpTime]))
VindTime.induced_voltage_sum()

# Third, calculate the induced voltage based on InducedVoltageResonator
atLineDensity = False #induced voltage is calculated at the line density?
if atLineDensity:
    tmpRes = InducedVoltageResonator(beam,slices,imp_source)
    tmpRes.induced_voltage_1turn()
    VindRes = TotalInducedVoltage(beam,slices,np.array([tmpRes]))
    VindRes.induced_voltage_sum()
else: #specify a custom array where the induced voltage is calculated
    time_array = np.linspace(VindFreq.time_array[0],\
                             1.5*VindFreq.time_array[-1],\
                             4*len(VindFreq.time_array))
    tmpRes = InducedVoltageResonator(beam,slices,imp_source,time_array)
    tmpRes.induced_voltage_1turn()

# Finally, calculate the analytic result
#Time array where to compute the induced voltage
gauss_time = np.linspace(VindFreq.time_array[0],1.5*VindFreq.time_array[-1],\
                4*len(VindFreq.time_array))
VindGauss = np.zeros(len(gauss_time))
#inducedVoltageGauss applies only to a single resonator; we have to sum over all
#resonators
for r in range(len(Q)):
#Notice that the time-argument is shifted by 
#mean(slices.bin_centers), because the analytical equation assumes the Gauss to
#be centered at t=0, but the line density is centered at 
#mean(slices.bin_centers)
    tmp = analytical_gaussian_resonator(bunch_sigma,Q[r],R[r],omega_r[r],\
                                gauss_time - np.mean(slices.bin_centers), \
                                beam.intensity)
    VindGauss += tmp.real 

# Plot the result of all four methods
# For times in slices.bin_centers all methods agree (i.e. curves overlap). For larger times, VindFreq and VindTime are not defined. 
plt.clf()
plt.ylabel("induced voltage [V]")
plt.xlabel("time [ns]")
plt.plot(1e9*VindFreq.time_array,VindFreq.induced_voltage)
plt.plot(1e9*VindFreq.time_array,VindTime.induced_voltage)
if atLineDensity:
    plt.plot(1e9*VindFreq.time_array,VindRes.induced_voltage)
else:
    plt.plot(1e9*time_array,tmpRes.induced_voltage)
plt.plot(1e9*gauss_time,VindGauss)
plt.show()

# Quantitatively compare the methods
#postion of maximum of VindTime.induced_voltage
pos = np.argmax(VindTime.induced_voltage)
#calculate the analytical value
VindGaussMax = 0.0
for r in range(len(Q)):
    tmp = analytical_gaussian_resonator(bunch_sigma,Q[r],R[r],omega_r[r],\
                    VindFreq.time_array[pos] - np.mean(slices.bin_centers), \
                    beam.intensity)
    VindGaussMax += tmp.real
#calculate the induced voltage with the semi-analytic method
tmpResMax =  InducedVoltageResonator(beam,slices,imp_source, \
                                     np.array([VindFreq.time_array[pos]]))
tmpResMax.induced_voltage_1turn()

#print the result
print("Ratio of induced voltage at the maximum of VindTime compared to the analytical result for the Frequency-, Time-, and Semi-Analytic method:")
print(VindFreq.induced_voltage[pos]/VindGaussMax,VindFreq.induced_voltage[pos]/VindGaussMax,tmpResMax.induced_voltage[0]/VindGaussMax)
print("Values should be:")
print(0.967417396592,0.967417396592,0.958548178272)
