import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

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

def project_S3_to_R3(S3):
    x, y, z, w = S3.T  # Transpose S3 before unpacking
    return x/(1-w), y/(1-w), z/(1-w)

"""--------------------main-----------------"""

num_points_on_great_circle = 32
num_points_on_great_arcs = 32
num_points_on_small_arcs = 16
num_points_on_interior_arcs = 8
m = 5
j = 2
B_surface = True
B_interiror = True

# Initialize the figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

### Generate, project and plot the great circle D ###
# Define two non-orthogonal vectors for the great circle
u_D = np.array([1, 1, 0, 0]) / np.sqrt(2)
v_D = np.array([0, 0, 1, 1]) / np.sqrt(2)
# Generate the great circle
D = generate_great_circle(num_points_on_great_circle, u_D, v_D)
# Project the great circle to 3D space
DX, DY, DZ = project_S3_to_R3(D)
ax.scatter(DX, DY, DZ, color='blue', s=1)  # Plot the great circle D in green
logging.debug("Great circle D generetad, projected and plottet")


### Generate, project and plot the complementary great circle E ###
# Generate the complementary great circle by rotating u and v by 90 degrees
u_E = np.array([-v_D[1], v_D[0], -v_D[3], v_D[2]])
v_E = np.array([-u_D[1], u_D[0], -u_D[3], u_D[2]])
# Generate the complementary great circle
E = generate_great_circle(num_points_on_great_circle, u_E, v_E)
# Project the complementary great circle to 3D space
EX, EY, EZ = project_S3_to_R3(E)
ax.scatter(EX, EY, EZ, color='red', s=1)  # Plot the complementary great circle E in red
logging.debug("Complementary great circle E generetad, projected and plottet")

### Subdivide D into m parts and selecting the border points. 
### Selecting the j-th vertex ###
# Calculate the step size
m_step_size = len(D) // m
# Select m points from D
d_points = [D[i * m_step_size % len(D)] for i in range(m)]
# select points on the edge connecting the points j and j+1
edge_points = D[j*m_step_size % len(D):(j+1)*m_step_size % len(D)]
logging.debug("Great circle D divided into m parts and border points selected")

# Project the m points to 3D space and plot
for d in d_points:
    dX, dY, dZ = project_S3_to_R3(d)
    # Add the points as big yellow dots dots on the plot
    ax.scatter(dX, dY, dZ, color='black', s=10)  # Plot the point from D
logging.debug("m points projected and plottet")

"""
# Project the edge points to 3D space and plot
for d in edge_points:
    dX, dY, dZ = project_S3_to_R3(d)
    # Add the points as big yellow dots dots on the plot
    ax.scatter(dX, dY, dZ, color='yellow', s=10)  # Plot the point from D
logging.debug("Edge points projected and plottet")
"""

### Choosing 8 points on the complementary Circle to span sceleton of the disk
# Calculate the step size
step_size = len(E) // 8
# Select 8 equally distant points
e_points = [E[i * step_size % len(D)] for i in range(8)]
logging.debug("8 sceleton points on the complementary Circle generetad")

### Generate the fat arcs, representing the sceleton of the disk
### Generate the small arcs, representing the Disk interior
if B_surface:
    for i in range(2):
        d = d_points[(j+i) % m]
        arcs = [generate_great_arc(num_points_on_great_arcs, d, p, np.arccos(np.dot(d, p))) for p in e_points]
        # Plot the points of the great arcs
        for arc in arcs:
            arc_X, arc_Y, arc_Z = project_S3_to_R3(arc)
            ax.scatter(arc_X, arc_Y, arc_Z, color='black', s=0.8)  # Plot the new great circle A in black
        # Generate the small arcs, representing the Disk interior
        small_arcs = [generate_great_arc(num_points_on_small_arcs, d, p, np.arccos(np.dot(d, p))) for p in E]
        for arc in small_arcs:
            arc_X, arc_Y, arc_Z = project_S3_to_R3(arc)
            ax.scatter(arc_X, arc_Y, arc_Z, color='black', s=0.5)  # Plot the new great circle A in black
    logging.debug("Boundary Disks generetad, projected and plottet")

### Generate the interior of the 3-Ball by connecting the edge points with the points on the complementary circle
if B_interiror:
    for e in edge_points:
        arcs = [generate_great_arc(num_points_on_interior_arcs, e, p, np.arccos(np.dot(e, p))) for p in E]
        # Plot the points of the interiro
        norm = np.linalg.norm(arcs[np.random.randint(0, len(arcs))][np.random.randint(0, num_points_on_interior_arcs)])
        logging.debug("Norm of random Ball interior points: %s", norm)
        for arc in arcs:
            arc_X, arc_Y, arc_Z = project_S3_to_R3(arc)
            ax.scatter(arc_X, arc_Y, arc_Z, color='purple', s=0.1, alpha=0.5)  # Plot the new great circle A in black


# Set the limits of the plot to -2 to 2 in all directions
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.grid(False) 

plt.show()