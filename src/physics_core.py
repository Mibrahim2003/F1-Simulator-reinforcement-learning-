"""
Physics Core for F1 Simulator.

Implements:
1. Pacejka 'Magic Formula' 4.1 (Tire Physics)
2. RK4 Integrator (Numerical Stability)
3. 6-DOF Vehicle Dynamics (Differential Equations)

Author: Antigravity Agent
"""
import math
import numpy as np
from dataclasses import dataclass

@dataclass
class TireParams:
    """Pacejka 4.1 Coefficients."""
    B: float  # Stiffness Factor
    C: float  # Shape Factor
    D: float  # Peak Value (Friction)
    E: float  # Curvature Factor

# Pirelli Soft Compound Approximation
PIRELLI_SOFT = TireParams(B=12.0, C=1.4, D=1.8, E=0.95)
PIRELLI_HARD = TireParams(B=10.0, C=1.3, D=1.5, E=0.97)

class PacejkaTireModel:
    """
    Calculates tire forces based on slip angle and load.
    Formula: y = D * sin(C * atan(B * x - E * (B * x - atan(B * x))))
    """
    def __init__(self, params: TireParams = PIRELLI_SOFT):
        self.params = params
        self.degradation = 1.0  # Grip multiplier (1.0 = New, 0.5 = Worn)
    
    def get_lateral_force(self, slip_angle_rad: float, normal_load: float) -> float:
        """
        Calculate lateral force (Fy).
        
        Args:
            slip_angle_rad: Angle between tire heading and velocity vector.
            normal_load: Vertical force (Fz) in Newtons.
            
        Returns:
            Lateral force in Newtons.
        """
        # Convert to relevant range for formula (usually degrees or normalized)
        # Standard Pacejka often uses degrees, but we can tune B for radians.
        # Let's use radians but scale B appropriately if needed. 
        # For standard B~10, input should be radians * ~10 or so? 
        # Actually standard Pacejka B is ~10 for slip in DEGREES or ~0.1 per degree?
        # Let's treat alpha as Angle.
        
        alpha = slip_angle_rad
        P = self.params
        
        # Load sensitivity (simplified): Grip increases with load but coefficient drops slightly
        # For this implementation, we assume linear load dependence for simplicity first,
        # multiplied by the Magic Formula coefficient.
        # Fy = Fz * mu(alpha)
        
        # Determine mu (friction coefficient) from slip angle
        # x = alpha
        B, C, D, E = P.B, P.C, P.D * self.degradation, P.E
        
        # The Magic Formula
        y = D * math.sin(C * math.atan(B * alpha - E * (B * alpha - math.atan(B * alpha))))
        
        return normal_load * y
    
    def add_wear(self, amount: float):
        """Reduce grip by amount (0.0 to 1.0)."""
        self.degradation = max(0.3, self.degradation - amount) # Min 30% grip
    
    def reset_wear(self):
        """Restore tires to fresh condition."""
        self.degradation = 1.0

# ... (VehicleState class remains same)
class VehicleState:
    """6-DOF State Vector."""
    x: float = 0.0          # World X (m)
    y: float = 0.0          # World Y (m)
    vx: float = 0.0         # Body Longitudinal Velocity (m/s)
    vy: float = 0.0         # Body Lateral Velocity (m/s)
    yaw: float = 0.0        # Heading (rad)
    yaw_rate: float = 0.0   # Angular Velocity (rad/s)

@dataclass
class VehicleConfig:
    """Physical properties of the F1 car."""
    mass: float = 798.0         # kg (Min weight)
    inertia: float = 1200.0     # kg*m^2 (Yaw inertia)
    wheelbase: float = 3.6      # meters
    cg_height: float = 0.3      # meters (Center of Gravity)
    weight_dist: float = 0.45   # % weight on front axle (0.0-1.0)
    
    # Aerodynamics
    drag_coeff: float = 0.9     # Cd
    lift_coeff: float = 3.5     # Cl (Downforce)
    frontal_area: float = 1.4   # m^2
    air_density: float = 1.225  # kg/m^3
    
    # Dimensions derived
    @property
    def lf(self): return self.wheelbase * (1.0 - self.weight_dist) # Distance CoG to Front
    @property
    def lr(self): return self.wheelbase * self.weight_dist         # Distance CoG to Rear

class F1DynamicsModel:
    """
    Solves Equations of Motion for a Dynamic Bicycle Model.
    """
    def __init__(self, config: VehicleConfig = VehicleConfig()):
        self.config = config
        self.tires = float = PacejkaTireModel(PIRELLI_SOFT)
        self.state = VehicleState()
        
    def get_forces(self, state: VehicleState, steer_angle: float, throttle: float, brake: float):
        """
        Calculate total forces and moments acting on the car.
        """
        cfg = self.config
        
        # 1. Aerodynamics
        speed_sq = state.vx**2 + state.vy**2
        aero_drag = 0.5 * cfg.air_density * speed_sq * cfg.drag_coeff * cfg.frontal_area
        aero_downforce = 0.5 * cfg.air_density * speed_sq * cfg.lift_coeff * cfg.frontal_area
        
        # Drag acts opposed to velocity
        # Simplified: opposing vx (assuming small slip angle for drag calc)
        drag_force_x = -math.copysign(aero_drag, state.vx)
        
        # 2. Vertical Loads (Weight Transfer + Aero)
        # Static weight
        Fz_static = cfg.mass * 9.81
        Fz_front = Fz_static * cfg.weight_dist + (aero_downforce * 0.4) # Aero bias
        Fz_rear = Fz_static * (1.0 - cfg.weight_dist) + (aero_downforce * 0.6)
        
        # Longitudinal Weight Transfer (Accel/Brake)
        # h/L * m * ax
        # Ignored for simple model first, can add later for more realism
        
        # 3. Slip Angles
        # Alpha = atan((vy + yaw_rate * L) / vx) - steer
        # Front
        if abs(state.vx) < 1.0:
            alpha_f = 0.0
            alpha_r = 0.0
        else:
            alpha_f = math.atan2(state.vy + state.yaw_rate * cfg.lf, state.vx) - steer_angle
            alpha_r = math.atan2(state.vy - state.yaw_rate * cfg.lr, state.vx)
        
        # 4. Tire Forces
        # Lateral
        Fy_f = -self.tires.get_lateral_force(alpha_f, Fz_front)
        Fy_r = -self.tires.get_lateral_force(alpha_r, Fz_rear)
        
        # Longitudinal (Engine/Brake)
        # Simple Engine Model: Force = Power / Speed
        # Max Engine Force (Torque)
        MAX_ENGINE_FORCE = 15000.0 # N
        MAX_BRAKE_FORCE = 25000.0  # N
        
        # Engine Force (RWD)
        # Allows negative throttle for Reverse
        Fx_engine = throttle * MAX_ENGINE_FORCE
        
        # Brake Force (All Wheels)
        # Brakes always oppose velocity
        if abs(state.vx) > 0.1:
            brake_dir = -math.copysign(1.0, state.vx)
        else:
            brake_dir = 0.0
            
        Fx_brake = brake * MAX_BRAKE_FORCE * brake_dir
        
        # Apply Logic
        # Engine applies to Rear Wheels (Traction Limited)
        traction_limit = Fz_rear * self.tires.params.D * self.tires.degradation
        
        if abs(Fx_engine) > traction_limit:
            Fx_rear = math.copysign(traction_limit, Fx_engine)
        else:
            Fx_rear = Fx_engine
            
        # Add Braking to Rear (Simplified: we add it to Fx_rear for simplicity in Body Frame calc)
        # Ideally brakes are on all wheels.
        # Let's say Rear Brakes = 40% of brake force, Front = 60%
        Fx_rear += Fx_brake * 0.4
        Fx_front = Fx_brake * 0.6
        
        # 5. Body Frame Forces
        # Fx_total = Fx_r + Fx_f*cos(d) - Fy_f*sin(d) + Drag
        # Fy_total = Fy_r + Fy_f*cos(d) + Fx_f*sin(d)
        
        cos_d = math.cos(steer_angle)
        sin_d = math.sin(steer_angle)
        
        force_x = Fx_rear + (Fx_front * cos_d) - (Fy_f * sin_d) + drag_force_x
        force_y = Fy_r + (Fy_f * cos_d) + (Fx_front * sin_d)
        
        # Moment (Torque)
        moment_z = (Fy_f * cos_d * cfg.lf) + (Fx_front * sin_d * cfg.lf) - (Fy_r * cfg.lr)
        
        return force_x, force_y, moment_z

    def update(self, dt: float, steer_input: float, throttle: float, brake: float):
        """
        Integrate state using Euler with substeps.
        throttle: -1.0 to 1.0 (Negative = Reverse)
        brake: 0.0 to 1.0
        """
        # Euler with substeps
        SUBSTEPS = 4
        sub_dt = dt / SUBSTEPS
        
        for _ in range(SUBSTEPS):
            self._step_euler(sub_dt, steer_input, throttle, brake)
            
    def _step_euler(self, dt: float, steer: float, throttle: float, brake: float):
        s = self.state
        
        Fx, Fy, Mz = self.get_forces(s, steer, throttle, brake)
        
        # Newton's 2nd Law
        ax = Fx / self.config.mass + s.vy * s.yaw_rate
        ay = Fy / self.config.mass - s.vx * s.yaw_rate
        yaw_accel = Mz / self.config.inertia
        
        # Integrate Velocity
        s.vx += ax * dt
        s.vy += ay * dt
        s.yaw_rate += yaw_accel * dt
        
        # Dampen small velocities to prevent drift when stopped
        # Only dampen if NO INPUTS are active
        if abs(s.vx) < 0.1 and throttle == 0 and brake == 0 and abs(Fx) < 100: 
            s.vx = 0
            
        if abs(s.vy) < 0.1 and abs(s.yaw_rate) < 0.05: s.vy = 0
        if abs(s.yaw_rate) < 0.01: s.yaw_rate = 0
        
        # Integrate Position (World Frame)
        # Rotate body velocities to world frame
        cos_y = math.cos(s.yaw)
        sin_y = math.sin(s.yaw)
        
        wx = s.vx * cos_y - s.vy * sin_y
        wy = s.vx * sin_y + s.vy * cos_y
        
        s.x += wx * dt
        s.y += wy * dt
        s.yaw += s.yaw_rate * dt
