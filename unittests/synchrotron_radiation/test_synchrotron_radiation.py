# coding: utf8
# Copyright 2014-2020 CERN. This software is distributed under the
# terms of the GNU General Public Licence version 3 (GPL Version 3),
# copied verbatim in the file LICENCE.md.
# In applying this licence, CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.
# Project website: http://blond.web.cern.ch/

"""
Unittest for synchrotron_radiation.synchrotron_radiation.py

:Authors: **Markus Schwarz**
"""

import unittest
import numpy as np
from scipy.constants import epsilon_0, e, m_e, c

from blond.utils import bmath as bm
from blond.input_parameters.ring import Ring
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.beam import Beam, Positron
from blond.beam.distributions import bigaussian
from blond.synchrotron_radiation.synchrotron_radiation import SynchrotronRadiation


class TestSynchtrotronRadiation(unittest.TestCase):
    
    
    # Run before every test
    def setUp(self):
        circumference = 110.4  # [m]
        energy = 2.5e9  # [eV]
        alpha = 0.0082
        self.R_bend = 5.559  # bending radius [m]
        C_gamma = e**2 / (3*epsilon_0 * (m_e*c**2)**4)  # [m J^3]
        C_gamma *= e**3  # [m eV^3]
  
        harmonic_number = 184
        voltage = 800e3  # eV
        phi_offsets = 0
       
        self.seed = 1234        
        intensity = 2.299e9
        n_macroparticles = int(1e4)
        sigma_dt = 10e-12  # RMS, [s]
        
        self.ring = Ring(circumference, alpha, energy, Positron(),
                    synchronous_data_type='total energy', n_turns=1)

        self.rf_station = RFStation(self.ring, harmonic_number, voltage,
                                    phi_offsets, n_rf=1)

        self.beam = Beam(self.ring, n_macroparticles, intensity)

        bigaussian(self.ring, self.rf_station, self.beam, sigma_dt, seed=self.seed)
        
        # energy loss per turn [eV]; assuming isomagnetic lattice
        self.U0 = C_gamma * self.ring.beta[0]**3 * self.ring.energy[0,0]**4 / self.R_bend

    def test_initial_beam(self):
        np.testing.assert_almost_equal(
            [self.beam.dt[0], self.beam.dt[-1]],
            [1.0054066581358374e-09, 9.981322445127657e-10], decimal=10,
            err_msg='Initial beam.dt wrong')
        np.testing.assert_almost_equal(
            [self.beam.dE[0], self.beam.dE[-1]],
            [337945.02937447827, -193066.62344453152], decimal=10,
            err_msg='Initial beam.dE wrong')

    def test_affect_only_dE(self):
        # incoherent synchrotron radiation, do displacement of beam
        iSR = SynchrotronRadiation(self.ring, self.rf_station, self.beam, self.R_bend,
                                   seed=self.seed, n_kicks=1, shift_beam=False,
                                   python=True, quantum_excitation=False)
        iSR.track()
        np.testing.assert_almost_equal(
            self.beam.dt[0], 1.0054066581358374e-09, decimal=10,
            err_msg='SR affected beam.dt')

    # def test_synchrotron_radiation_python(self):
    #     # incoherent synchrotron radiation
    #     iSR = SynchrotronRadiation(self.ring, self.rf_station, self.beam, self.R_bend,
    #                                n_kicks=2,
    #                                python=False, quantum_excitation=False,
    #                                seed=self.seed)
        
    
if __name__ == '__main__':

    unittest.main()
