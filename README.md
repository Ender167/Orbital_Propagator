# Overview
A CUDA-based propagator for space debris orbiting Earth based on the Symplectic Euler integrator. Objects are propagated independently. Problem is modeled as embarrassingly parallelizable.

The result is a binary file containing the position and relative energy error of each object for every n propagation steps.

# Features
* Support for the following factors that affect each object's position and velocity: Earth's central gravity, J2 perturbation;
* 2 CUDA kernels supporting single and double precision (float/double) to assess tradeoff between execution speed and accuracy;
* An OpenMP implementation running on the CPU supporting single precision meant to highlight the speedup provided by multithreading and to compare the GPU implementation;
* A sequential implementation running on the CPU supporting single precision, used as the base algorithm for measuring the performance increase provided by the parallel implementations;
* A Python-based visualization tool;

# Metrics
Performance metrics are used to assess the impact of parallelization in terms of execution speed. The final result writing step is further parallelized using OpenMP across all implementations.

Accuracy metrics measure fidelity of the simulation to real-world physics. Energy error represents the difference between the computed energy of an object at propagation step n and its base energy.

In order to evaluate the propagator, the projects offers access to the following pre-computed metrics:
* Performance: propagation step execution time (kernel execution time for CUDA), final results writing time (in milliseconds), speedup (ratio between base algorithm execution time and parallel implementation execution time);
* Accuracy: maximum energy error over the entire simulation, RMS energy error over the entire simulation, maximum energy error at final propagation step, mean final error at final propagation step;

# How to run
This project requires a Windows operating system, python 3.10 or later and an NVIDIA GPU with CUDA support. Run python main.py from Python Visualization.

# Showcase

<img width="1276" height="948" alt="Simulation_Image" src="https://github.com/user-attachments/assets/87542f21-0393-4101-83c4-f630ea6e8efa" />
<img width="745" height="533" alt="Main_UI" src="https://github.com/user-attachments/assets/f0093cea-2df0-4c4a-bfee-464261afdff0" />
