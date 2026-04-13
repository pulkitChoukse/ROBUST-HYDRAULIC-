from moc_module import PenstockParams, SimulationEngine

import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
import numpy as np


# --- Input Ranges ---
ranges = {
    "Penstock Length (m)": (100, 5000),
    "Pipe Diameter (m)": (0.5, 10),
    "Wave Speed (m/s)": (500, 2000),
    "Initial Velocity (m/s)": (0.5, 10),
    "Initial Pressure Head (m)": (50, 500),
    "Max Pressure Head (m)": (100, 600),
    "Min Pressure Head (m)": (0, 100),
    "Closure Time (s)": (1, 60)
}

# Fixed friction factor (constant for all runs)
FIXED_FRICTION = 0.015


# --- Simulation Function ---
def run_simulation():
    try:
        # Get user inputs
        values = []
        for label, entry in zip(labels, entries):
            val = float(entry.get())
            min_val, max_val = ranges[label]
            if not (min_val <= val <= max_val):
                messagebox.showerror("Range Error", f"{label} must be between {min_val} and {max_val}.")
                return
            values.append(val)

        (length, diameter, wave_speed, velocity,
         init_head, max_head, min_head,
         closure_time) = values

        # Create penstock parameters
        params = PenstockParams(
            length=length,
            diameter=diameter,
            wave_speed=wave_speed,
            initial_velocity=velocity,
            initial_pressure_head=init_head,
            max_pressure_head=max_head,
            min_pressure_head=min_head,
            friction_factor=FIXED_FRICTION,  # constant friction
            n_segments=100
        )

        if not params.validate():
            messagebox.showerror("Validation Error", "Invalid penstock parameters.")
            return

        # Run simulation
        engine = SimulationEngine()
        result = engine.run_simulation(params, closure_time)

        # Safety check
        if result.is_safe:
            status_label.config(text=f"Safe. Peak Head = {result.head:.2f} m", fg="green")
        else:
            status_label.config(text=f"Unsafe! Peak Head = {result.head:.2f} m", fg="red")

        # Plot results
        fig, axs = plt.subplots(2, 1, figsize=(8, 6))

        axs[0].plot(result.time_arr, result.pressure_head, 'b-', label="Pressure Head")
        axs[0].set_title("Pressure vs Time")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Pressure Head (m)")
        axs[0].legend()

        axs[1].plot(result.time_arr, result.velocity, 'r-', label="Velocity")
        axs[1].set_title("Velocity vs Time")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Velocity (m/s)")
        axs[1].legend()

        plt.tight_layout()
        plt.show()

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")


def clear_inputs():
    for entry in entries:
        entry.delete(0, tk.END)
    status_label.config(text="")


# --- Tkinter GUI ---
root = tk.Tk()
root.title("Hydraulic Transient Analysis Dashboard")

# Frames
input_frame = tk.LabelFrame(root, text="Input Parameters", padx=10, pady=10)
input_frame.pack(padx=10, pady=10, fill="x")

button_frame = tk.Frame(root)
button_frame.pack(pady=5)

status_frame = tk.Frame(root)
status_frame.pack(pady=5)

# Input fields with updated defaults
labels = [
    "Penstock Length (m)", "Pipe Diameter (m)", "Wave Speed (m/s)",
    "Initial Velocity (m/s)", "Initial Pressure Head (m)",
    "Max Pressure Head (m)", "Min Pressure Head (m)",
    "Closure Time (s)"
]

defaults = ["00", "00", "00", "00", "00", "00", "00", "00"]

entries = []
for i, (label, default) in enumerate(zip(labels, defaults)):
    tk.Label(input_frame, text=f"{label} (Range: {ranges[label][0]} - {ranges[label][1]})").grid(row=i, column=0, sticky="w")
    entry = tk.Entry(input_frame)
    entry.insert(0, default)
    entry.grid(row=i, column=1)
    entries.append(entry)

(entry_length, entry_diameter, entry_wave_speed, entry_velocity,
 entry_init_head, entry_max_head, entry_min_head,
 entry_closure) = entries

# Buttons
tk.Button(button_frame, text="Run Simulation", command=run_simulation).pack(side="left", padx=5)
tk.Button(button_frame, text="Clear", command=clear_inputs).pack(side="left", padx=5)
tk.Button(button_frame, text="Exit", command=root.quit).pack(side="left", padx=5)

# Status label
status_label = tk.Label(status_frame, text="", font=("Arial", 12))
status_label.pack()

root.mainloop()
