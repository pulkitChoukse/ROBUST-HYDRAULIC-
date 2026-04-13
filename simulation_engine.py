#module for MOC guided water transinets inside simple single penstock(pipe with a valve)

import numpy as np

#class --->  PenstockParams (for input parameters ,  configuration  of the penstock variables )

class PenstockParams:
    """
    length : float
        L — Total length of the penstock in metres.

    diameter : float
        D — Internal diameter of the penstock in metres.

    wave_speed : float
        c — Acoustic wave speed (celerity) in m/s.
    
    initial_velocity : float
        V0 — Flow velocity in the penstock before the trip event, in m/s.
    
    initial_pressure_head : float
        H0 — The total piezometric head at the turbine inlet under

    
    max_pressure_head : float
        H_max — Maximum allowable piezometric head in metres.

    min_pressure_head : float
        H_min — Minimum allowable piezometric head in metres.
    
    friction_factor : float
        f — Darcy-Weisbach friction factor (dimensionless).

     n_segments : int
        N — Number of spatial segments the penstock is divided into.
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
       
         # ---> primary attributes 
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
        # ---> checks for validating teh penstocks parameters

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
    def to_feature_vector(self):  # to_feature_vector() --> gives input for ML model to predict

        # ---> returns an ordered 1d array of the parameters to do further clacs
        return np.array([
            self.length,
            self.diameter,
            self.wave_speed,
            self.initial_velocity,
            self.initial_pressure_head,
            self.max_pressure_head,
            self.min_pressure_head,
        ])




class SimulationEngine:

    # Executes the MOC simulation for a single penstock
    
    def __init__(self,sim_duration_factor=6.0):

        self.g=9.81     # gravitational acceleration [m/s²]
        self.sim_duration_factor =sim_duration_factor    # number of wave round-trips to simulate
    
    def run_simulation(self, params, closure_time):
        
        # Main entry point. 
        # Runs the full MOC loop and returns a SimulationResult.
        
        result = self._compute_wave_dynamics(params, closure_time)
        result.check_safety(params.max_pressure_head, params.min_pressure_head)
        return result
    
