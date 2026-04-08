#module for MOC guided water transinets inside simple single penstock(pipe with a valve)

import numpy as np

#class -- PenstockParams (for input parameters ,  configuration  of the penstock variables )

class PenstockParams:
    """
    length : float
        L — Total length of the penstock in metres.

    diameter : float
        D — Internal diameter of the penstock in metres.

    wave_speed : float
        c — Acoustic wave speed (celerity) in m/s.

    
    """

    def __init__(
        self,
        length,
        diameter,
        wave_speed,
        initial_velocity,
        initial_pressure_head,
        max_pressure_head,
        min_pressure_head,
        friction_factor=0.015,
        n_segments=100,
    ):
       
         # ── primary attributes  ────────────────────────
        self.length                = float(length)
        self.diameter              = float(diameter)
        self.wave_speed            = float(wave_speed)
        self.initial_velocity      = float(initial_velocity)
        self.initial_pressure_head = float(initial_pressure_head)
        self.max_pressure_head     = float(max_pressure_head)
        self.min_pressure_head     = float(min_pressure_head)
        self.friction_factor       = float(friction_factor)
        self.n_segments            = int(n_segments)

    def validate(self):
        #checks for validating teh penstocks parameters

        checks = [
            self.length > 0,
            self.diameter > 0,
            self.wave_speed > 0,
            self.initial_velocity > 0,
            self.initial_pressure_head > 0,
            self.max_pressure_head > self.initial_pressure_head,
            self.min_pressure_head < self.initial_pressure_head,
            self.friction_factor > 0,
            self.n_segments >= 10,
        ]
        return all(checks)
    def to_feature_vector(self):

        #the ordered 1d array of the parameters to do further clacs
        return np.array([
            self.length,
            self.diameter,
            self.wave_speed,
            self.initial_velocity,
            self.initial_pressure_head,
            self.max_pressure_head,
            self.min_pressure_head,
        ])

