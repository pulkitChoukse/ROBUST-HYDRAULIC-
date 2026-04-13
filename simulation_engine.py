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




"""
Helpers for simulation Engine to simulate motion by solving boundary conditions and interior nodes by MOC 
as private methods in SimulationEngine class in diagrams
here declared as independent functions to ease the testing
-- can also be defined as private inside SimulationEngine class 
"""

def _guide_vane_tau(t, closure_time):
    """
    Linear closure profile: τ = 1 at t=0 (fully open), 0 at t=Tc (closed).
    np.clip clamps the result so it never goes below 0 after Tc.
    """
    return float(np.clip(1.0 - t / closure_time, 0.0, 1.0))


def _upstream_bc(C_minus, B, H_reservoir):
    """
    Upstream (reservoir) boundary — constant head.
    H is fixed = H_reservoir; Q is derived from the C- characteristic.
    C- arrives from node 1: H_P = C_M + B*Q_P → Q_P = (H_res - C_M) / B
    """
    H_P = H_reservoir
    Q_P = (H_P - C_minus) / B
    return H_P, Q_P


def _downstream_bc(C_plus, B, tau, Q0, H0):
    """
    Downstream (guide vane) boundary — variable orifice.
    Orifice equation: Q = Kv*τ*√H, where Kv = Q0/√H0 (calibrated at steady state).
    Substituting into C+ characteristic gives a quadratic in u = √H:
        u² + (B*Kv*τ)*u - C_P = 0  →  solve for positive root.
    """
    Kv = Q0 / np.sqrt(max(H0, 1e-6))   # valve coefficient calibrated from initial conditions

    if tau <= 0.0:               # guide vane fully closed: dead-end reflection
        return C_plus, 0.0

    b = B * Kv * tau            # quadratic coefficient
    discriminant = b**2 + 4.0 * C_plus  # always ≥ 0 for physical inputs
    discriminant = max(discriminant, 0.0)

    u   = (-b + np.sqrt(discriminant)) / 2.0  # positive root for u = √H_P
    H_P = u**2
    Q_P = Kv * tau * u
    return H_P, Q_P


def _solve_interior(H_A, Q_A, H_B, Q_B, B, R):

    """
    Interior node is updated using both C+ and C- characteristics.
    C+ from left neighbour (i-1):  C_P = H_A + B*Q_A - R*Q_A*|Q_A|
    C- from right neighbour (i+1): C_M = H_B - B*Q_B + R*Q_B*|Q_B|
    Solving simultaneously: Q_P = (C_P - C_M)/(2B),  H_P = (C_P + C_M)/2
    abs(Q) in friction term because friction always opposes flow direction.
    """
    C_P = H_A + B * Q_A - R * Q_A * abs(Q_A)
    C_M = H_B - B * Q_B + R * Q_B * abs(Q_B)
    Q_P = (C_P - C_M) / (2.0 * B)
    H_P = (C_P + C_M) / 2.0
    return H_P, Q_P




# Main class that uses the params to simulate and return SimulationResult object as output 

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
    
