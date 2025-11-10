import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import argparse
import os
import logging
try:
    import imageio
except ImportError:
    imageio = None  # We'll warn later if GIF export is requested without imageio

# Set up logging (suppress DEBUG by default)
logging.basicConfig(level=logging.INFO)
# Silence chatty third-party debug logs
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)

def generate_great_circle(num_points, u, v):
    # Generate t values
    t = np.linspace(0, 2*np.pi, num_points)

    # Compute the x, y, z, and w coordinates of the points on the great circle
    x = np.cos(t) * u[0] + np.sin(t) * v[0]
    y = np.cos(t) * u[1] + np.sin(t) * v[1]
    z = np.cos(t) * u[2] + np.sin(t) * v[2]
    w = np.cos(t) * u[3] + np.sin(t) * v[3]

    # Return a single array of shape (num_points, 4)
    return np.array([x, y, z, w]).T

def generate_great_arc(n, u, v, phi):
    """Generate n points on the great arc defined by vectors u and v and angle phi."""
    # Generate the points on the great arc
    theta = np.linspace(0, phi, n)  # Only go up to pi/2 to get the arc
    arc = np.array([np.cos(t) * u + np.sin(t) * v for t in theta])

    return arc

def construct_3_ball():
    ### Generate, project and plot the great circle D ###
    # Define two non-orthogonal vectors for the great circle
    u_D = np.array([1, 1, 0, 0]) / np.sqrt(2)
    v_D = np.array([0, 0, 1, 1]) / np.sqrt(2)
    # Generate the great circle
    D = generate_great_circle(num_points_on_great_circle, u_D, v_D)
    logging.debug("Great circle D generetad.")

    ### Generate, project and plot the complementary great circle E ###
    # Generate the complementary great circle by rotating u and v by 90 degrees
    u_E = np.array([-v_D[1], v_D[0], -v_D[3], v_D[2]])
    v_E = np.array([-u_D[1], u_D[0], -u_D[3], u_D[2]])
    # Generate the complementary great circle
    E = generate_great_circle(num_points_on_great_circle, u_E, v_E)

    ### Subdivide D into m parts and selecting the border points. 
    ### Selecting the j-th vertex ###
    # Calculate the step size
    m_step_size = len(D) // m
    # Select m points from D
    d_points = [D[i * m_step_size % len(D)] for i in range(m)]
    # select points on the edge connecting the points j and j+1
    edge_points = D[j*m_step_size % len(D):(j+1)*m_step_size % len(D)]
    logging.debug("Great circle D divided into m parts and border points selected")

    ### Choosing 8 points on the complementary Circle to span sceleton of the disk
    # Calculate the step size
    step_size = len(E) // 8
    # Select 8 equally distant points
    e_points = [E[i * step_size % len(D)] for i in range(8)]
    logging.debug("8 sceleton points on the complementary Circle generetad")

    surface_arcs, small_surface_arcs, interior_arcs = archer(E, d_points, edge_points, e_points)

    return D, E, d_points, edge_points, e_points, surface_arcs, small_surface_arcs, interior_arcs

def archer(E, d_points, edge_points, e_points):
    ### Generate the fat arcs, representing the sceleton of the disk
    ### Generate the small arcs, representing the Disk interior

    if B_surface:
        # Use skeleton points e_points for both sets so the visible structure changes as d_points shift
        surface_arcs = [generate_great_arc(num_points_on_great_arcs, d, p, np.arccos(np.dot(d, p))) for p in e_points for d in d_points[j:j+2]]
        small_surface_arcs = [generate_great_arc(num_points_on_small_arcs, d, p, np.arccos(np.dot(d, p))) for p in e_points for d in d_points[j:j+2]]
        logging.debug("Boundary Disks generetad, projected and plottet")
    
    ### Generate the interior of the 3-Ball by connecting the edge points with the points on the complementary circle
    if B_interiror:
        interior_arcs = [generate_great_arc(num_points_on_interior_arcs, e, p, np.arccos(np.dot(e, p))) for p in E for e in edge_points]
    elif not B_interiror:
        interior_arcs = []
        logging.debug("No interior arcs generated")
    
    return surface_arcs, small_surface_arcs, interior_arcs

def turn_three_ball(three_ball, step=1):
    D, E, d_points, edge_points, e_points, surface_arcs, small_surface_arcs, interior_arcs = three_ball
    m_step_size = len(D) // m # Calculate the step size
    shift_d_points = [D[((i * m_step_size) + step) % len(D)] for i in range(m)] # Select m points from D and shift by 1
    edge_points = D[((j * m_step_size) + step) % len(D):(j+1)*m_step_size % len(D)]

    surface_arcs, small_surface_arcs, interior_arcs = archer(E, shift_d_points, edge_points, e_points)

    # Return shift_d_points to keep tuple consistent with regenerated arcs
    return D, E, shift_d_points, edge_points, e_points, surface_arcs, small_surface_arcs, interior_arcs


def project_S3_to_R3(S3):
    x, y, z, w = S3.T  # Transpose S3 before unpacking
    return x/(1-w), y/(1-w), z/(1-w)


def projplo_3B(three_ball, ax):
    """Project and plot the current 3-ball configuration without extra global rotation."""

    D, E, d_points, edge_points, e_points, surface_arcs, small_surface_arcs, interior_arcs = three_ball

    DX, DY, DZ = project_S3_to_R3(D)
    D_plot = ax.scatter(DX, DY, DZ, color='blue', s=1)  # Plot the great circle D in green

    # Project the complementary great circle to 3D space
    EX, EY, EZ = project_S3_to_R3(E)
    E_plot = ax.scatter(EX, EY, EZ, color='red', s=1)  # Plot the complementary great circle E in red

    logging.debug("Great circle D divided into m parts and border points selected")

    # Project the m points to 3D space and plot

    d_points_plot = [ax.scatter(project_S3_to_R3(d)[0], project_S3_to_R3(d)[1], project_S3_to_R3(d)[2], color='black', s=10) for d in d_points]
    logging.debug("m points projected and plottet")


    if B_surface:
        # Generate the fat arcs, representing the sceleton of the disk
        surface_arcs_plot = [ax.scatter(project_S3_to_R3(arc)[0], project_S3_to_R3(arc)[1], project_S3_to_R3(arc)[2], color='black', s=0.8) for arc in surface_arcs]  # Plot the new great circle A in black
        # Generate the small arcs, representing the Disk interior
        small_surface_arcs_plot = [ax.scatter(project_S3_to_R3(arc)[0], project_S3_to_R3(arc)[1], project_S3_to_R3(arc)[2], color='black', s=0.5) for arc in small_surface_arcs]  # Plot the new great circle A in black
        logging.debug("Boundary Disks projected and plottet")
    elif not B_surface:
        small_surface_arcs_plot = []
        logging.debug("No boundary Disks projected and plottet")

    if B_interiror:
        ### Generate the interior of the 3-Ball by connecting the edge points with the points on the complementary circle
        interior_arcs_plot = [ax.scatter(project_S3_to_R3(arc)[0], project_S3_to_R3(arc)[1], project_S3_to_R3(arc)[2], color='purple', s=0.1, alpha=0.5) for arc in interior_arcs]  # Plot the new great circle A in black
        logging.debug("Interior Disks projected and plottet")
    elif not B_interiror:
        interior_arcs_plot = []
        logging.debug("No interior Disks projected and plottet")

    return [D_plot, E_plot, d_points_plot, surface_arcs_plot, small_surface_arcs_plot, interior_arcs_plot]

"""-------------------- CLI / main -----------------"""

# Default parameters (can be overridden via CLI)
num_points_on_great_circle = 64
num_points_on_great_arcs = 32
num_points_on_small_arcs = 16
num_points_on_interior_arcs = 32
m = 5
j = 2
B_surface = True
B_interiror = True
resolution = 1

def run_animation(steps, pause=0.1, show=True):
    """Run an interactive animation using plt.pause (legacy behavior)."""
    three_ball = construct_3_ball()
    for i in range(steps):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        projplo_3B(three_ball, ax)
        three_ball = turn_three_ball(three_ball, i+1)
        if show:
            plt.pause(pause)
        plt.close(fig)
    if show:
        plt.show()

def export_gif(output_path="animation.gif", steps=None, fps=10):
    """Generate a GIF by rendering each frame to an image.

    Parameters
    ----------
    output_path : str
        Destination GIF filename.
    steps : int | None
        Number of frames; defaults to len(edge_points)+1 if None.
    fps : int
        Frames per second in resulting GIF.
    """
    if imageio is None:
        raise RuntimeError("imageio is not installed; install it (pip install imageio) to use GIF export.")

    # Reconstruct base geometry once
    three_ball = construct_3_ball()
    edge_points_len = len(three_ball[3])
    if steps is None:
        steps = edge_points_len + 1

    frame_dir = "frames"
    os.makedirs(frame_dir, exist_ok=True)

    frame_paths = []
    logging.info(f"Generating {steps} frames for GIF...")
    for i in range(steps):
        fig = plt.figure(figsize=(4,4))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)
        ax.set_zlim(-2, 2)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_box_aspect((1,1,1))
        projplo_3B(three_ball, ax)
        frame_file = os.path.join(frame_dir, f"frame_{i:04d}.png")
        fig.savefig(frame_file, dpi=150, bbox_inches='tight')
        frame_paths.append(frame_file)
        plt.close(fig)
        three_ball = turn_three_ball(three_ball, i+1)

    # Read frames and write GIF
    # Use imageio.v2.imread when available to avoid deprecation warnings
    imread = getattr(imageio, "v2", imageio).__getattribute__("imread") if imageio is not None else None
    images = [imread(fp) for fp in frame_paths]
    duration = 1.0 / fps  # seconds per frame
    imageio.mimsave(output_path, images, duration=duration)
    logging.info(f"GIF saved to {output_path} ({len(images)} frames @ {fps} fps)")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Visualize S^3 great circles and export animation.")
    parser.add_argument("--gif", action="store_true", help="Export animation as GIF instead of interactive display.")
    parser.add_argument("--gif-path", default="animation.gif", help="Output path for GIF (default: animation.gif)")
    parser.add_argument("--steps", type=int, default=None, help="Number of frames; default uses edge_points length + 1")
    parser.add_argument("--fps", type=int, default=10, help="Frames per second for GIF")
    parser.add_argument("--no-surface", action="store_true", help="Disable boundary surface arcs")
    parser.add_argument("--no-interior", action="store_true", help="Disable interior arcs")
    parser.add_argument("--interactive", action="store_true", help="Force interactive legacy animation even if --gif is set")

    args = parser.parse_args()

    global B_surface, B_interiror
    if args.no_surface:
        B_surface = False
    if args.no_interior:
        B_interiror = False

    if args.gif and not args.interactive:
        export_gif(output_path=args.gif_path, steps=args.steps, fps=args.fps)
    else:
        # Run legacy interactive animation
        # Determine steps (edge_points length + 1)
        temp = construct_3_ball()
        steps_default = len(temp[3]) + 1
        steps = args.steps or steps_default
        run_animation(steps=steps, pause=0.1, show=True)

if __name__ == "__main__":
    main()
