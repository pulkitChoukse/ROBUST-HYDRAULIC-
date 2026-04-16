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


        # --> derived geometry (computed from above, not user input)
        self.area              = np.pi * self.diameter**2 / 4.0   # A = πD²/4
        self.initial_discharge = self.initial_velocity * self.area # Q0 = V0·A
        self.dx                = self.length / self.n_segments     # spatial step
        self.dt                = self.dx / self.wave_speed         # CFL time step
        self.B                 = self.wave_speed / (9.81 * self.area)  # impedance c/(gA)
        # friction resistance per segment: R = f·Δx / (2g·D·A²)
        self.R = (self.friction_factor * self.dx) / (2.0 * 9.81 * self.diameter * self.area**2)
        

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


# simulationEngine generates series of velocity and pressure as  output which is returned as SimulationResult object 
# this class is used to better handle the Output of simulation from Simulation Engine


class SimulationResult:
    """
    Stores all outputs from one MOC simulation.

    """

    def __init__(self, time_arr, pressure_head, velocity, closure_time, wave_period):
        self.time_arr      = time_arr       # 1D array — time axis [s]
        self.pressure_head = pressure_head  # 1D array — head at turbine end [m]
        self.velocity      = velocity       # 1D array — velocity at turbine end [m/s]
        self.closure_time  = closure_time   # Tc used for this run [s]
        self.wave_period   = wave_period    # 2L/c [s]
        self.head          = float(np.max(pressure_head))  # peak head — "head" in domain model
        self.min_head      = float(np.min(pressure_head))
        self.is_safe       = False          # set by check_safety()
        self.n_steps       = len(time_arr)
        self.time_range    = f"[0..{time_arr[-1]:.1f}] s"

    def check_safety(self, max_pressure_head, min_pressure_head):
        """
        Evaluates whether peak and min head stay within limits.
        Sets self.is_safe and returns it.

        This helps in displaying Safe or Unsafe warns about on dashboard .
        """
        self.is_safe = (
            self.head     <= max_pressure_head and
            self.min_head >= min_pressure_head
        )
        return self.is_safe

    def to_dict(self):
        """
        return result as a plain dict for the Dashboard to use to plot on graphs.
        """
        return {
            "time":          self.time_arr,
            "pressure_head": self.pressure_head,
            "velocity":      self.velocity,
            "peak_head":     self.head,
            "min_head":      self.min_head,
            "is_safe":       self.is_safe,
            "closure_time":  self.closure_time,
            "wave_period":   self.wave_period,
            "n_steps":       self.n_steps,
            "time_range":    self.time_range,
        }

    def visualize_outputs(self):
        """
        can be delacred here also  for visualising
        but the visualisation part of output or plotting is handles by dashboard 
        """
        pass



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
    

    def _compute_wave_dynamics(self, params, closure_time):
        
        """
        main MOC time-stepping loop.
        
        Returns a SimulationResult with raw arrays before safety check.
        """
        
        p  = params
        N  = p.n_segments
        B  = p.B
        R  = p.R
        Q0 = p.initial_discharge
        H0 = p.initial_pressure_head

        wave_period = 2.0 * p.length / p.wave_speed      # 2L/c
        t_end       = self.sim_duration_factor * wave_period
        n_steps     = int(t_end / p.dt) + 1

        # initial steady-state at head of penstock (linear, head drops due to friction)
        total_friction = R * N * Q0 * abs(Q0)
        H_reservoir    = H0 + total_friction

        H = np.array([H_reservoir - (i / N) * total_friction for i in range(N + 1)])
        Q = np.full(N + 1, Q0)   # uniform discharge at steady state

        # output storage
        time_arr      = np.zeros(n_steps)
        pressure_head = np.zeros(n_steps)
        Q_out         = np.zeros(n_steps)

        time_arr[0]      = 0.0
        pressure_head[0] = H[N]
        Q_out[0]         = Q[N]

        H_new = np.zeros(N + 1)
        Q_new = np.zeros(N + 1)

        for step in range(1, n_steps):
            t   = step * p.dt
            tau = _guide_vane_tau(t, closure_time)  # current vane opening fraction(0-1)

            # interior nodes i = 1 … N-1
            for i in range(1, N):
                H_new[i], Q_new[i] = _solve_interior(
                    H[i - 1], Q[i - 1],
                    H[i + 1], Q[i + 1],
                    B, R
                )

            # upstream BC: C- from node 1 → node 0
            C_M_up          = H[1] - B * Q[1] + R * Q[1] * abs(Q[1])
            H_new[0], Q_new[0] = _upstream_bc(C_M_up, B, H_reservoir)

            # downstream BC: C+ from node N-1 → node N
            C_P_dn          = H[N - 1] + B * Q[N - 1] - R * Q[N - 1] * abs(Q[N - 1])
            H_new[N], Q_new[N] = _downstream_bc(C_P_dn, B, tau, Q0, H0)

            H[:] = H_new
            Q[:] = Q_new

            time_arr[step]      = t
            pressure_head[step] = H[N]
            Q_out[step]         = Q[N]

        velocity = Q_out / p.area   # V = Q/A

        return SimulationResult(time_arr, pressure_head, velocity, closure_time, wave_period)


    def joukowsky_rise(self, params):
        """
        Theoretical upper-bound pressure rise for instantaneous closure.
        Formula: ΔH = c*V0/g  (Joukowsky 1898).

        Returns (delta_H, peak_H).
        """
        delta_H = params.wave_speed * params.initial_velocity / self.g
        peak_H  = params.initial_pressure_head + delta_H
        return delta_H, peak_H

    
# the code for a test case for analysing its working

if __name__ == "__main__":
    print("=" * 58)
    print("SimulationEngine — test case")
    print("=" * 58)

    params = PenstockParams(
        length                = 1000.0,
        diameter              = 2.0,
        wave_speed            = 1000.0,
        initial_velocity      = 3.0,
        initial_pressure_head = 200.0,
        max_pressure_head     = 280.0,
        min_pressure_head     = 20.0,
        friction_factor       = 0.015,
        n_segments            = 100,
    )

    print(f"\nparams.validate() = {params.validate()}")
    print(f"feature vector = {params.to_feature_vector()}")

    engine   = SimulationEngine()
    dH, peak = engine.joukowsky_rise(params)
    print(f"\nJoukowsky rise : ΔH = {dH:.1f} m  →  peak = {peak:.1f} m")
    print(f"Wave period    : 2L/c = {2*params.length/params.wave_speed:.2f} s")

    result = engine.run_simulation(params, closure_time=15.0)
    print(f"\nrun_simulation(Tc=15s)")
    print(f"  n_steps   = {result.n_steps}")
    print(f"  peak head = {result.head:.1f} m  (limit {params.max_pressure_head} m)")
    print(f"  min head  = {result.min_head:.1f} m  (limit {params.min_pressure_head} m)")
    print(f"  is_safe   = {result.is_safe}")
    print(f"  to_dict() keys = {list(result.to_dict().keys())}")

    print("\tTest-case finished.")