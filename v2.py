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

num_points_on_great_circle = 64
num_points_on_great_arcs = 32
num_points_on_small_arcs = 16

logging.debug("Generating great circle D")
# Define two non-orthogonal vectors for the great circle
u_D = np.array([1, 1, 0, 0]) / np.sqrt(2)
v_D = np.array([0, 0, 1, 1]) / np.sqrt(2)
# Generate the great circle
D = generate_great_circle(num_points_on_great_circle, u_D, v_D)
logging.debug("Great circle D generated")

logging.debug("Projecting great circle D to 3D space")  
# Project the great circle to 3D space
DX, DY, DZ = project_S3_to_R3(D)
logging.debug("Great circle D projected to 3D space")

logging.debug("Generating complementary great circle E")
u_E = np.array([-v_D[1], v_D[0], -v_D[3], v_D[2]])
v_E = np.array([-u_D[1], u_D[0], -u_D[3], u_D[2]])

# Generate the complementary great circle
E = generate_great_circle(num_points_on_great_circle, u_E, v_E)
logging.debug("Complementary great circle E generated")

logging.debug("Projecting complementary great circle E to 3D space")
# Project the complementary great circle to 3D space
EX, EY, EZ = project_S3_to_R3(E)
logging.debug("Complementary great circle E projected to 3D space")

# Plot the projected points
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(DX, DY, DZ, color='green', s=1)  # Plot the great circle D in green
ax.scatter(EX, EY, EZ, color='red', s=1)  # Plot the complementary great circle E in red

logging.debug("Selecting point on D")

# Select one point on D at random 
d_index = np.random.randint(len(D))
d = np.array([D[d_index][0], D[d_index][1], D[d_index][2], D[d_index][3]])
# Project the points to 3D space
dX, dY, dZ = project_S3_to_R3(d)
# Add the points as big green dots on the plot
ax.scatter(dX, dY, dZ, color='yellow', s=10)  # Plot the point from D

logging.debug("Selected point on D: %s", d)
logging.debug("Generating great arcs")


# Calculate the step size
step_size = len(E) // 8
# Select 8 equally distant points
e_points = [E[i * step_size % len(D)] for i in range(8)]

logging.debug("e_points norm: %s", np.linalg.norm(e_points[0]))
logging.debug("dot product of d and e_points : %s", [np.dot(d, p) for p in e_points])

points_X, points_Y, points_Z = project_S3_to_R3(np.array(e_points)) # Project the points to 3D space
ax.scatter(points_X, points_Y, points_Z, color='blue', s=10)  # Plot the points on E in blue
arcs = [generate_great_arc(num_points_on_great_arcs, d, p, np.arccos(np.dot(d, p))) for p in e_points]

# Plot the points of the great arcs
for arc in arcs:
    arc_X, arc_Y, arc_Z = project_S3_to_R3(arc)
    ax.scatter(arc_X, arc_Y, arc_Z, color='black', s=0.8)  # Plot the new great circle A in black

logging.debug("norm of random point on arc: %s", np.linalg.norm(arcs[np.random.randint(len(arcs))][np.random.randint(len(arcs[0]))]))
logging.debug("Great arcs generated")

logging.debug("Generating small arcs")
# Generate the small arcs, representing the Disk
small_arcs = [generate_great_arc(num_points_on_small_arcs, d, p, np.arccos(np.dot(d, p))) for p in E]
for arc in small_arcs:
    arc_X, arc_Y, arc_Z = project_S3_to_R3(arc)
    ax.scatter(arc_X, arc_Y, arc_Z, color='black', s=0.1)  # Plot the new great circle A in black

logging.debug("Small arcs generated")


# Set the limits of the plot to -2 to 2 in all directions
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.grid(False) 

plt.show()