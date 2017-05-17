
# Copyright 2016 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

'''
SPS simulation with intensity effects in time and frequency domains as well 
as the semi-analytic method using a table of resonators. The input beam has 
been cloned to show that the three methods are equivalent (compare the three 
figure folders). Note that to create an exact clone of the beam, the option 
seed=1 in the generation has been used. After the tracking all induced voltages 
are plotted, together with the analytic result of a Gaussian bunch.
This script shows also an example of how to use the class SliceMonitor (check
the corresponding h5 files).
'''

from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

from input_parameters.general_parameters import GeneralParameters
from input_parameters.rf_parameters import RFSectionParameters
from trackers.tracker import RingAndRFSection
from beams.beams import Beam
from beams.distributions import longitudinal_bigaussian
from monitors.monitors import BunchMonitor
from beams.slices import Slices
from impedances.impedance import InducedVoltageTime, InducedVoltageFreq
from impedances.impedance import InducedVoltageResonator, TotalInducedVoltage
from impedances.induced_voltage_analytical import analytical_gaussian_resonator
from impedances.impedance_sources import Resonators
from plots.plot_beams import *
from plots.plot_impedance import *
from plots.plot_slices import *
from plots.plot import Plot


# SIMULATION PARAMETERS -------------------------------------------------------

# Beam parameters
particle_type = 'proton'
n_particles = 1e10      
n_macroparticles = 5*1e6
tau_0 = 2e-9 # [s]

# Machine and RF parameters
gamma_transition = 1/np.sqrt(0.00192)   # [1]
C = 6911.56  # [m]
      
# Tracking details
n_turns = 2          
dt_plt = 1          

# Derived parameters
sync_momentum = 25.92e9 # [eV / c]
momentum_compaction = 1 / gamma_transition**2 # [1]       

# Cavities parameters
n_rf_systems = 1                                     
harmonic_number = 4620                         
voltage_program = 0.9e6 # [V]
phi_offset = 0.0

# DEFINE RING------------------------------------------------------------------

general_params = GeneralParameters(n_turns, C, momentum_compaction,
                                   sync_momentum, particle_type)
general_params_freq = GeneralParameters(n_turns, C, momentum_compaction,
                                        sync_momentum, particle_type)
general_params_res = GeneralParameters(n_turns, C, momentum_compaction,
                                        sync_momentum, particle_type)


RF_sct_par = RFSectionParameters(general_params, n_rf_systems, harmonic_number, 
                          voltage_program, phi_offset)
RF_sct_par_freq = RFSectionParameters(general_params_freq, n_rf_systems,
                                      harmonic_number, voltage_program,
                                      phi_offset)
RF_sct_par_res = RFSectionParameters(general_params_res, n_rf_systems,
                                      harmonic_number, voltage_program,
                                      phi_offset)

my_beam = Beam(general_params, n_macroparticles, n_particles)
my_beam_freq = Beam(general_params_freq, n_macroparticles, n_particles)
my_beam_res = Beam(general_params_res, n_macroparticles, n_particles)

ring_RF_section = RingAndRFSection(RF_sct_par, my_beam)
ring_RF_section_freq = RingAndRFSection(RF_sct_par_freq, my_beam_freq)
ring_RF_section_res = RingAndRFSection(RF_sct_par_res, my_beam_res)

# DEFINE BEAM------------------------------------------------------------------

longitudinal_bigaussian(general_params, RF_sct_par, my_beam, tau_0/4, seed=1)
longitudinal_bigaussian(general_params_freq, RF_sct_par_freq, my_beam_freq,
                        tau_0/4, seed=1)
longitudinal_bigaussian(general_params_res, RF_sct_par_res, my_beam_res,
                        tau_0/4, seed=1)

number_slices = 2**8
slice_beam = Slices(RF_sct_par, my_beam, number_slices, cut_left=0,
                    cut_right=2*np.pi, cuts_unit='rad', fit_option='gaussian')
slice_beam_freq = Slices(RF_sct_par_freq, my_beam_freq, number_slices,
                         cut_left=0, cut_right=2 * np.pi, cuts_unit='rad',
                         fit_option='gaussian')
slice_beam_res = Slices(RF_sct_par_freq, my_beam_res, number_slices,
                         cut_left=0, cut_right=2 * np.pi, cuts_unit='rad',
                         fit_option='gaussian')

# MONITOR----------------------------------------------------------------------

bunchmonitor = BunchMonitor(general_params, ring_RF_section, my_beam, 
                            '../output_files/TC5_output_data',
                            Slices=slice_beam, buffer_time=1)

bunchmonitor_freq = BunchMonitor(general_params_freq, ring_RF_section_freq,
                         my_beam_freq, '../output_files/TC5_output_data_freq',
                         Slices=slice_beam_freq, buffer_time=1)
bunchmonitor_res = BunchMonitor(general_params_res, ring_RF_section_res,
                         my_beam_res, '../output_files/TC5_output_data_res',
                         Slices=slice_beam_res, buffer_time=1)


# LOAD IMPEDANCE TABLE--------------------------------------------------------

table = np.loadtxt('../input_files/TC5_new_HQ_table.dat', comments = '!')

R_shunt = table[:, 2] * 10**6 
f_res = table[:, 0] * 10**9
Q_factor = table[:, 1]
resonator = Resonators(R_shunt, f_res, Q_factor)

ind_volt_time = InducedVoltageTime(my_beam, slice_beam, [resonator])
ind_volt_freq = InducedVoltageFreq(my_beam_freq, slice_beam_freq, [resonator],
                                   1e5)
ind_volt_res = InducedVoltageResonator(my_beam_res,slice_beam_res,resonator)

tot_vol = TotalInducedVoltage(my_beam, slice_beam, [ind_volt_time])
tot_vol_freq = TotalInducedVoltage(my_beam_freq, slice_beam_freq,
                                   [ind_volt_freq])
tot_vol_res = TotalInducedVoltage(my_beam_res, slice_beam_res,
                                   [ind_volt_res])

# Analytic result-----------------------------------------------------------
VindGauss = np.zeros(len(slice_beam.bin_centers))
for r in range(len(Q_factor)):
#Notice that the time-argument of inducedVoltageGauss is shifted by 
#mean(my_slices.bin_centers), because the analytical equation assumes the
#Gauss to be centered at t=0, but the line density is centered at 
#mean(my_slices.bin_centers)
    tmp = analytical_gaussian_resonator(tau_0/4, \
                    Q_factor[r],R_shunt[r],2*np.pi*f_res[r], \
                    slice_beam.bin_centers - np.mean(slice_beam.bin_centers), \
                    my_beam.intensity)
    VindGauss += tmp.real 

# PLOTS

format_options = {'dirname': '../output_files/TC5_fig/1', 'linestyle': '.'}
plots = Plot(general_params, RF_sct_par, my_beam, dt_plt, n_turns, 0, 
             0.0014*harmonic_number, -1.5e8, 1.5e8, xunit='rad',
             separatrix_plot=True, Slices=slice_beam,
             h5file='../output_files/TC5_output_data', 
             histograms_plot=True, sampling=50, format_options=format_options)

format_options = {'dirname': '../output_files/TC5_fig/2', 'linestyle': '.'}
plots_freq = Plot(general_params_freq, RF_sct_par_freq, my_beam_freq, dt_plt,
                  n_turns, 0, 0.0014*harmonic_number, -1.5e8, 1.5e8,
                  xunit='rad', separatrix_plot=True, Slices=slice_beam_freq, 
                  h5file='../output_files/TC5_output_data_freq', 
                  histograms_plot=True, sampling=50,
                  format_options=format_options)
format_options = {'dirname': '../output_files/TC5_fig/3', 'linestyle': '.'}
plots_res = Plot(general_params_res, RF_sct_par_res, my_beam_res, dt_plt,
                  n_turns, 0, 0.0014*harmonic_number, -1.5e8, 1.5e8,
                  xunit='rad', separatrix_plot=True, Slices=slice_beam_res, 
                  h5file='../output_files/TC5_output_data_res', 
                  histograms_plot=True, sampling=50,
                  format_options=format_options)


# ACCELERATION MAP-------------------------------------------------------------

map_ = [tot_vol] + [ring_RF_section] + [slice_beam] + [bunchmonitor] + [plots]
map_freq = [tot_vol_freq] + [ring_RF_section_freq] + [slice_beam_freq] \
    + [bunchmonitor_freq] + [plots_freq]
map_res = [tot_vol_res] + [ring_RF_section_res] + [slice_beam_res] \
    + [bunchmonitor_res] + [plots_res]

# TRACKING + PLOTS-------------------------------------------------------------

for i in np.arange(1, n_turns+1):
    
    print(i)
    for m in map_:
        m.track()
    for m in map_freq:
        m.track()
    for m in map_res:
        m.track()
    
    # Plots
    if (i % dt_plt) == 0:
        plot_induced_voltage_vs_bin_centers(i, general_params, tot_vol,
                                style='.', dirname='../output_files/TC5_fig/1')
        plot_induced_voltage_vs_bin_centers(i, general_params_freq,
                  tot_vol_freq, style='.', dirname='../output_files/TC5_fig/2')
        plot_induced_voltage_vs_bin_centers(i, general_params_res,
                  tot_vol_res, style='.', dirname='../output_files/TC5_fig/3')

# Plotting induced voltages---------------------------------------------------
plt.figure(0)
plt.clf()
plt.ylabel("induced voltage [arb. unit]")
plt.xlabel("time [ns]")
plt.plot(1e9*slice_beam.bin_centers,tot_vol.induced_voltage,label='Time')
plt.plot(1e9*slice_beam_freq.bin_centers,tot_vol_freq.induced_voltage,\
         label='Freq')
plt.plot(1e9*slice_beam_res.bin_centers,tot_vol_res.induced_voltage,\
    label='Resonator')
plt.plot(1e9*slice_beam.bin_centers,VindGauss,label='Analytic')
plt.legend()
plt.show()
plt.pause(0.0001)

print("Done!")
