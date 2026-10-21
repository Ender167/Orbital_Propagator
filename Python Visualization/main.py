
import glob
import os
import re
import numpy as np
import pyvista as pv
import time
import struct
import tkinter as tk
import multiprocessing
import subprocess

CUDA_EXE = r"E:\Projects\Disertation\CUDA_Object_Tracking\Object_Tracking\x64\Release\Object_Tracking.exe"

EARTH_RADIUS = 6371.0
POINT_SIZE = 3
FRAME_DELAY = 0.03

def numerical_sort(value):
    match = re.search(r'frame_(\d+)\.csv', value)
    return int(match.group(1)) if match else -1

def preprocessFiles():
    folder = "./frames"
    files = glob.glob(os.path.join(folder, "frame_*.csv"))
    files = sorted(files, key=numerical_sort)

    if not files:
        raise ValueError("No CSV files found.")

    return files

def preprocessBinary(binFile = "framesBinary/simulation.bin"):
    with open(binFile, "rb") as f:
        header = f.read(8)
        num_frames, N = struct.unpack("ii", header)

        frames = []
        for _ in range(num_frames):
            t = struct.unpack("i", f.read(4))[0]
            data = np.fromfile(f, dtype=np.float32, count=3 * N)
            frames.append(data.reshape(N, 3))

        return frames
def renderOrbit(files, binary=0, plotter1=None):
    if binary == 0:
        data = np.loadtxt(files[0], delimiter=",", skiprows=1)
    else:
        data = files[0]
    points = data[:, :3]

    cloud = pv.PolyData(points)

    if plotter1 is None:
        plotter = pv.Plotter()
    else:
        plotter = plotter1

    plotter.set_background("black")

    text_actor = plotter.add_text("", position="upper_left", font_size=12, color="white")

    # Earth
    texture = pv.read_texture("earth.jpg")
    earth = pv.Sphere(radius=EARTH_RADIUS)
    earth.texture_map_to_sphere(inplace=True)

    earth_actor = plotter.add_mesh(earth, texture=texture) #color="royalblue"

    # Equatorial plane
    plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1),
                     i_size=20000, j_size=20000)

    # Stars
    num_stars = 2000
    if num_stars > 0:
        stars = np.random.uniform(-50000, 50000, (num_stars, 3))

        star_cloud = pv.PolyData(stars)

        plotter.add_points(
            star_cloud,
            color="white",
            point_size=1,
            opacity=0.05
        )
    #

    plotter.add_mesh(plane, color="gray", opacity=0.1)
    # Debris
    r = np.linalg.norm(points, axis=1)
    cloud["altitude"] = r

    plotter.add_points(
        cloud,
        scalars="altitude",
        cmap="viridis",
        point_size=POINT_SIZE,
        scalar_bar_args={
            "title": "Altitude (km)",
            "color": "white",
            "title_font_size": 16,
            "label_font_size": 12,
            "fmt": "%.5f"
        }
    )
    #

    plotter.camera_position = 'iso'
    plotter.show(auto_close=False, interactive_update=True)

    # TRAIL LOGIC
    trail_length = 20
    history = []
    trail_cloud = pv.PolyData(points)

    plotter.add_points(
        trail_cloud,
        color="cyan",
        point_size=2,
        opacity=0.3,
        render_points_as_spheres=False
    )
    #

    for i, f in enumerate(files):
        if binary == 0:
            data = np.loadtxt(f, delimiter=",", skiprows=1)
        else:
            data = f
        points = data[:, :3]

        cloud.points = points
        cloud["altitude"] = np.linalg.norm(points, axis=1)

        earth_actor.RotateZ(0.2)

        # TRAIL LOGIC
        history.append(points.copy())
        if len(history) > trail_length:
            history.pop(0)

        trail_points = np.vstack(history)
        trail_cloud.points = trail_points
        #
        altitudes = np.linalg.norm(points, axis=1)
        text_actor.SetText(
            0,
            f"Frame: {i + 1}/{len(files)}\n"
            f"Objects: {len(points)}\n"
            f"Mean altitude: {altitudes.mean():.0f} km"
        )

        plotter.update()
        time.sleep(FRAME_DELAY)

    #plotter.show()

def run_simulation_cuda():
    files = preprocessBinary()
    renderOrbit(files, 1)

def run_simulation_omp():
    files = preprocessBinary("framesBinary/simulation_omp.bin")
    renderOrbit(files, 1)

def run_simulation_seq():
    files = preprocessBinary("framesBinary/simulation_seq.bin")
    renderOrbit(files, 1)

def run_cuda():
    precision = precision_to_int(precision_var.get())
    version_to_run = 2
    subprocess.Popen([CUDA_EXE, str(version_to_run), str(precision)])

def run_omp():
    precision = precision_to_int(precision_var.get())
    version_to_run = 1
    subprocess.Popen([CUDA_EXE, str(version_to_run), str(precision)])

def run_seq():
    precision = precision_to_int(precision_var.get())
    version_to_run = 0
    subprocess.Popen([CUDA_EXE, str(version_to_run), str(precision)])

def start_simulation_cuda():
    p = multiprocessing.Process(target=run_simulation_cuda)
    p.start()
def start_simulation_omp():
    p = multiprocessing.Process(target=run_simulation_omp)
    p.start()
def start_simulation_seq():
    p = multiprocessing.Process(target=run_simulation_seq)
    p.start()

def precision_to_int(precision_str):
    mapping = {"float": 0, "double": 1}
    return mapping.get(precision_str, 0)

def main():
    root = tk.Tk()
    root.title("Orbital Propagation Simulator")
    root.geometry("600x400")

    top_frame = tk.Frame(root)
    top_frame.pack(pady=15)

    tk.Label(top_frame, text="Precision:").pack(side=tk.LEFT, padx=5)
    global precision_var
    precision_var = tk.StringVar(root)
    precision_var.set("float")

    precision_dropdown = tk.OptionMenu(top_frame, precision_var, "float", "double")
    precision_dropdown.pack(side=tk.LEFT, padx=5)

    frame1 = tk.Frame(root)
    frame1.pack(pady=15)

    tk.Button(
        frame1,
        text="Run CUDA Kernel",
        command=run_cuda,
        width=22,
        height=2
    ).pack(side=tk.LEFT, padx=5)

    tk.Button(
        frame1,
        text="Run OMP",
        command=run_omp,
        width=22,
        height=2
    ).pack(side=tk.LEFT, padx=5)

    tk.Button(
        frame1,
        text="Run Sequential",
        command=run_seq,
        width=22,
        height=2
    ).pack(side=tk.LEFT, padx=5)

    frame2 = tk.Frame(root)
    frame2.pack(pady=15)

    tk.Button(
        frame2,
        text="Start Simulation - CUDA",
        command=start_simulation_cuda,
        width=22,
        height=2
    ).pack(side=tk.LEFT, padx=5)

    tk.Button(
        frame2,
        text="Start Simulation - OpenMP",
        command=start_simulation_omp,
        width=22,
        height=2
    ).pack(side=tk.LEFT, padx=5)

    tk.Button(
        frame2,
        text="Start Simulation - Sequential",
        command=start_simulation_seq,
        width=22,
        height=2
    ).pack(side=tk.LEFT, padx=5)

    root.mainloop()


#def main():
#    files = preprocessBinary()
#    renderOrbit(files, 1)
if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
