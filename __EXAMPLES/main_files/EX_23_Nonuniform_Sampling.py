#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:33:30 2020

@author: MarkusArbeit
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e

# BLonD imports
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.beam import Beam, Proton
from blond.beam.sparse_slices import SparseSlices
from blond.beam.profile import Profile, CutOptions
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage
from blond.impedances.induced_voltage_analytical import analytical_gaussian_resonator
from blond.impedances.impedance_sources import Resonators
# from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.beam.distributions import bigaussian
# from impedance_scenario import scenario, impedance2blond

def FourierTransform(k, x, y):

    res = np.zeros_like(k, dtype=np.complex)
    
    kappa = np.diff(y) / np.diff(x)
    
    if type(k) == np.ndarray:
        indexes = k != 0
        phase = np.exp(-1j * np.outer(k[indexes],x) )
    
        phase_diff = np.diff(phase, axis=1)
    
        res[indexes] = np.sum(kappa * phase_diff, axis=1) / k[indexes]**2
        
        np.invert(indexes, indexes)
        # res[indexes] = -0.5 * np.sum( np.diff(x**2) * kappa)
        res[indexes] = -0.5 * np.sum( (x[:-1] + x[1:]) * np.diff(y) )

    else:  # k is a single number
        if k == 0.0:
            res = -0.5 * np.sum( (x[:-1] + x[1:]) * np.diff(y) )
        else:
            phase = np.exp(-1j * k * x )
    
            phase_diff = np.diff(phase)
    
            res = np.sum(kappa * phase_diff) / k**2
    
    return res


def sample_function(func, points, tol=0.05, min_points=16, max_level=16,
                    sample_transform=None):
    """
    Sample a 1D function to given tolerance by adaptive subdivision.

    The result of sampling is a set of points that, if plotted,
    produces a smooth curve with also sharp features of the function
    resolved.

    Parameters
    ----------
    func : callable
        Function func(x) of a single argument. It is assumed to be vectorized.
    points : array-like, 1D
        Initial points to sample, sorted in ascending order.
        These will determine also the bounds of sampling.
    tol : float, optional
        Tolerance to sample to. The condition is roughly that the total
        length of the curve on the (x, y) plane is computed up to this
        tolerance.
    min_point : int, optional
        Minimum number of points to sample.
    max_level : int, optional
        Maximum subdivision depth.
    sample_transform : callable, optional
        Function w = g(x, y). The x-samples are generated so that w
        is sampled.

    Returns
    -------
    x : ndarray
        X-coordinates
    y : ndarray
        Corresponding values of func(x)

    Notes
    -----
    This routine is useful in computing functions that are expensive
    to compute, and have sharp features --- it makes more sense to
    adaptively dedicate more sampling points for the sharp features
    than the smooth parts.

    Examples
    --------
    >>> def func(x):
    ...     '''Function with a sharp peak on a smooth background'''
    ...     a = 0.001
    ...     return x + a**2/(a**2 + x**2)
    ...
    >>> x, y = sample_function(func, [-1, 1], tol=1e-3)

    >>> import matplotlib.pyplot as plt
    >>> xx = np.linspace(-1, 1, 12000)
    >>> plt.plot(xx, func(xx), '-', x, y[0], '.')
    >>> plt.show()

    """
    return _sample_function(func, points, values=None, mask=None, depth=0,
                            tol=tol, min_points=min_points, max_level=max_level,
                            sample_transform=sample_transform)

def _sample_function(func, points, values=None, mask=None, tol=0.05,
                     depth=0, min_points=16, max_level=16,
                     sample_transform=None):
    points = np.asarray(points)

    if values is None:
        values = np.atleast_2d(func(points))

    if mask is None:
        mask = Ellipsis

    if depth > max_level:
        # recursion limit
        print('Warning: Maximum recursion reached')
        return points, values

    x_a = points[...,:-1][mask]
    x_b = points[...,1:][mask]

    x_c = .5*(x_a + x_b)
    y_c = np.atleast_2d(func(x_c))

    x_2 = np.r_[points, x_c]
    y_2 = np.r_['-1', values, y_c]
    j = np.argsort(x_2)

    x_2 = x_2[...,j]
    y_2 = y_2[...,j]

    # -- Determine the intervals at which refinement is necessary

    if len(x_2) < min_points:
        mask = np.ones([len(x_2)-1], dtype=bool)
    else:
        # represent the data as a path in N dimensions (scaled to unit box)
        if sample_transform is not None:
            y_2_val = sample_transform(x_2, y_2)
        else:
            y_2_val = y_2

        p = np.r_['0',
                  x_2[None,:],
                  y_2_val.real.reshape(-1, y_2_val.shape[-1]),
                  y_2_val.imag.reshape(-1, y_2_val.shape[-1])
                  ]

        sz = (p.shape[0]-1)//2

        xscale = x_2.ptp(axis=-1)
        yscale = abs(y_2_val.ptp(axis=-1)).ravel()

        p[0] /= xscale
        p[1:sz+1] /= yscale[:,None]
        p[sz+1:]  /= yscale[:,None]

        # compute the length of each line segment in the path
        dp = np.diff(p, axis=-1)
        s = np.sqrt((dp**2).sum(axis=0))
        s_tot = s.sum()

        # compute the angle between consecutive line segments
        dp /= s
        dcos = np.arccos(np.clip((dp[:,1:] * dp[:,:-1]).sum(axis=0), -1, 1))

        # determine where to subdivide: the condition is roughly that
        # the total length of the path (in the scaled data) is computed
        # to accuracy `tol`
        dp_piece = dcos * .5*(s[1:] + s[:-1])
        mask = (dp_piece > tol * s_tot)

        mask = np.r_[mask, False]
        mask[1:] |= mask[:-1].copy()


    # -- Refine, if necessary

    if mask.any():
        return _sample_function(func, x_2, y_2, mask, tol=tol, depth=depth+1,
                                min_points=min_points, max_level=max_level,
                                sample_transform=sample_transform)
    else:
        return x_2, y_2[0]

np.random.seed(1984)

intensity_pb = 1.0e11
sigma = 0.2e-9

n_macroparticles_pb = int(1e4)
n_bunches = 2

### --- Ring and RF ----------------------------------------------
intensity = n_bunches * intensity_pb     # total intensity SPS
n_turns = 1
# Ring parameters SPS
circumference = 6911.5038 # Machine circumference [m]
sync_momentum = 25.92e9 # SPS momentum at injection [eV/c]

gamma_transition = 17.95142852  # Q20 Transition gamma
momentum_compaction = 1./gamma_transition**2  # Momentum compaction array

ring = Ring(circumference, momentum_compaction, sync_momentum, Proton(),
            n_turns=n_turns)
tRev = ring.t_rev[0]

# RF parameters SPS 
n_rf_systems = 1    # Number of rf systems 

harmonic_number = 4620  # Harmonic numbers
voltage = 3.5e6  # [V]
phi_offsets = 0

rf_station = RFStation(ring, harmonic_number, voltage, phi_offsets, n_rf=1)
t_rf = rf_station.t_rf[0,0]

bunch_spacing = 5  # RF buckets

n_macroparticles = n_bunches * n_macroparticles_pb
beam = Beam(ring, n_macroparticles, intensity)

for bunch in range(n_bunches):
    
    bunchBeam = Beam(ring, n_macroparticles_pb, intensity_pb)
    bigaussian(ring, rf_station, bunchBeam, sigma, reinsertion=True, seed=1984+bunch)
    
    beam.dt[bunch*n_macroparticles_pb : (bunch+1)*n_macroparticles_pb] \
        = bunchBeam.dt + bunch*bunch_spacing * t_rf
    beam.dE[bunch*n_macroparticles_pb : (bunch+1)*n_macroparticles_pb] = bunchBeam.dE


### uniform profile

profile_margin = 0 * t_rf

t_batch_begin = 0 * t_rf
t_batch_end = (bunch_spacing * (n_bunches-1) + 1) * t_rf

n_slices_rf = 64 #number of slices per RF-bucket

cut_left = t_batch_begin - profile_margin
cut_right = t_batch_end + profile_margin

# number of rf-buckets of the beam 
# + rf-buckets before the beam + rf-buckets after the beam
n_slices    = n_slices_rf * (bunch_spacing * (n_bunches-1) + 1 \
            + int(np.round((t_batch_begin - cut_left)/t_rf)) \
            + int(np.round((cut_right - t_batch_end)/t_rf)))

uniform_profile = Profile(beam, CutOptions = CutOptions(cut_left=cut_left,
                                                cut_right=cut_right, n_slices=n_slices))
uniform_profile.track()


### impedance

R_s = np.asarray([1e6, 1e6])
f_r = np.asarray([200e6, 400e6])
Q = np.array([0.5, 4]) * np.pi*f_r * t_rf  # short and long range wake field

impedance_model = [Resonators(R_s, f_r, Q, method='python')]

# Induced voltage calculated by the 'frequency' method
# frequency_step = 4*ring.f_rev[0]
frequency_step = 1e6
uniform_frequency_object = InducedVoltageFreq(beam, uniform_profile,
                                              impedance_model, frequency_step)
# using automatic frequency resolution gives wrong result
# uniform_frequency_object = InducedVoltageFreq(beam, uniform_profile, mpedance_model)


induced_voltage = TotalInducedVoltage(beam, uniform_profile, [uniform_frequency_object])
induced_voltage.induced_voltage_sum()

# figf = plt.figure('freq domain', clear=True)
# axf = figf.gca()
# axf.grid()
# axf.plot(uniform_profile.beam_spectrum_freq/1e6, np.abs(uniform_profile.beam_spectrum), 'C0')
# axf2 = axf.twinx()
# axf2.plot(SPS_freq.freq/1e6, np.real(SPS_freq.total_impedance * SPS_freq.profile.bin_size) / 1e6, 'C1')

# figt = plt.figure('profile', clear=True)
# axt = figt.gca()
# axt.grid()
# axt.plot(uniform_profile.bin_centers*1e9, uniform_profile.n_macroparticles, '-')
# axt2 = axt.twinx()
# axt2.plot(induced_voltage.time_array*1e9, induced_voltage.induced_voltage / 1e6, 'C1')

filling_pattern = np.zeros(bunch_spacing * (n_bunches-1) + 1)
filling_pattern[::bunch_spacing] = 1

### sparse beam
nonuniform_profile = SparseSlices(rf_station, beam, n_slices_rf, filling_pattern,
                                  tracker='onebyone')
nonuniform_profile.track()

time = nonuniform_profile.bin_centers_array.flatten()
profile = nonuniform_profile.n_macroparticles_array.flatten()

# axt.plot(time*1e9, profile, '.')

Vind_anal = np.zeros_like(time)
for bunch in range(n_bunches):
    for it in range(len(R_s)):
        Vind_anal += analytical_gaussian_resonator(sigma, Q[it], R_s[it], 2*np.pi*f_r[it], 
                                                   time - (bunch*bunch_spacing+0.5) * t_rf,
                                                   intensity_pb)
# axt2.plot(time*1e9, Vind_anal / 1e6, '.')


def _compute_impedance(f, InducedVoltageFreqObject):
    InducedVoltageFreqObject.sum_impedances(f)
    return InducedVoltageFreqObject.total_impedance.real

omega = 2*np.pi * uniform_frequency_object.freq
# omega *= 2*np.pi
Lambda = FourierTransform(omega, time, profile / np.trapz(profile, time))
Z = uniform_frequency_object.total_impedance * uniform_frequency_object.profile.bin_size
# Z *= SPS_freq.profile.bin_size
Y = Z*Lambda
Vind_nonuni = -2*intensity_pb * e * FourierTransform(-time, omega, Y) / np.pi

nonuniform_frequency_object = InducedVoltageFreq(beam, uniform_profile, impedance_model)
freq, tmp = sample_function(lambda f: _compute_impedance(f, nonuniform_frequency_object), 
                            np.linspace(0,5e9,1000), tol=0.01)
nonuniform_frequency_object.sum_impedances(freq)
Z2 = nonuniform_frequency_object.total_impedance * nonuniform_frequency_object.profile.bin_size
# _tmp = SPS_freq.total_impedance
# freq, tmp = sample_function(lambda f: _compute_impedance(f, SPS_freq), 
#                           np.linspace(0,5e9,1000), tol=0.01)
# SPS_freq.sum_impedances(freq)
# Z2 = SPS_freq.total_impedance * SPS_freq.profile.bin_size
## Z2 *= SPS_freq.profile.bin_size
# SPS_freq.total_impedance = _tmp

Lambda2 = FourierTransform(2*np.pi*freq, time, profile / np.trapz(profile, time))
Y2 = Z2 * Lambda2
Vind_nonuni2 = -2*intensity_pb * e * FourierTransform(-time, 2*np.pi*freq, Y2).real / np.pi

print(f"Data points for uniform frequency object:\t\t {len(uniform_frequency_object.total_impedance)}")
print(f"Data points for nonuniform frequency object:\t {len(nonuniform_frequency_object.total_impedance)}")


#%%

plt.figure('profile', clear=True)
plt.grid()
plt.xlabel('time / ns')
plt.ylabel('macro-particles')
tmp, = plt.plot(uniform_profile.bin_centers*1e9, uniform_profile.n_macroparticles, '.',
                label='Profile, uniform')
for bunch in range(n_bunches):
    indexes = (time>nonuniform_profile.cut_left_array[bunch]) * (time<nonuniform_profile.cut_right_array[bunch])
    tmp2, = plt.plot(time[indexes]*1e9, profile[indexes], 'C1-', label='SparseSlices')
plt.legend(handles=[tmp, tmp2])
plt.tight_layout()

plt.figure('impedance', clear=True)
plt.grid()
plt.plot(uniform_frequency_object.freq / 1e6, 
         uniform_frequency_object.total_impedance.real * uniform_profile.bin_size / 1e6, '.')
plt.plot(freq / 1e6, Z2.real / 1e6)

plt.figure('integrand', clear=True)
plt.grid()
plt.plot(uniform_frequency_object.freq / 1e6,
         (uniform_frequency_object.total_impedance*uniform_profile.beam_spectrum).real
         * uniform_profile.bin_size / n_macroparticles)
plt.plot(freq / 1e6, Y2.real)
         

plt.figure('voltage', clear=True)
plt.grid()
plt.xlabel('time / ns')
plt.ylabel('induced voltage / MV')
tmp, = plt.plot(induced_voltage.time_array*1e9, induced_voltage.induced_voltage / 1e6, '.',
                label='uniform')
for bunch in range(n_bunches):
    indexes = (time>nonuniform_profile.cut_left_array[bunch]) * (time<nonuniform_profile.cut_right_array[bunch])

    tmp2, = plt.plot(time[indexes]*1e9, Vind_nonuni2[indexes] / 1e6, 'C1-', label='non-uniform')
    tmp3, = plt.plot(time[indexes]*1e9, Vind_anal[indexes] / 1e6, 'C2--', label='analytic')
plt.legend(handles=[tmp, tmp2, tmp3])
plt.tight_layout()
# plt.plot(time*1e9, Vind_nonuni / 1e6)


# plt.figure('profile', clear=True)
# plt.grid()
# plt.plot(induced_voltage.time_array*1e9, induced_voltage.induced_voltage / 1e6, '.')
# plt.plot(time*1e9, Vind_anal / 1e6)
# # plt.plot(time*1e9, Vind_nonuni / 1e6)
# plt.plot(time*1e9, Vind_nonuni2 / 1e6, '--')


# plt.figure('voltage', clear=True)
# plt.grid()
# plt.plot(induced_voltage.time_array*1e9, induced_voltage.induced_voltage / 1e6)
# plt.plot(time*1e9, Vind_anal / 1e6, '.')
# plt.plot(time*1e9, Vind_nonuni / 1e6)
# plt.plot(time*1e9, Vind_nonuni2 / 1e6)


# plt.figure('impedance', clear=True)
# plt.grid()
# plt.plot(SPS_freq.freq/1e6, np.imag(SPS_freq.total_impedance * SPS_freq.profile.bin_size) / 1e6, '.')
# plt.plot(omega/2/np.pi /1e6, Z.imag / 1e6)
# plt.plot(freq / 1e6, Z2.imag / 1e6, '--')

# plt.figure('spectrum', clear=True)
# plt.grid()
# plt.plot(uniform_profile.beam_spectrum_freq/1e6, np.imag(uniform_profile.beam_spectrum), '.')
# plt.plot(omega/2/np.pi/1e6, np.imag(Lambda) * n_macroparticles)
# plt.plot(freq/1e6, np.imag(Lambda2) * n_macroparticles, '--')

# plt.figure('integrand', clear=True)
# plt.grid()
# plt.plot(omega/2/np.pi/1e6, np.abs(Y))
# plt.plot(freq/1e6, np.abs(Y2), '.')

