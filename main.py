import numpy as np
import matplotlib.pyplot as plt

# Function to generate evenly distributed points on S^3 using spherical coordinates
def generate_points_on_s3(num_points):
    phi = np.arccos(2 * np.random.rand(num_points) - 1)
    theta = 2 * np.pi * np.random.rand(num_points)
    psi = 2 * np.pi * np.random.rand(num_points)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    w = np.cos(psi)
    
    points = np.vstack((x, y, z, w)).T
    return points

# Generate points on the 3-sphere
num_points = 64
points = generate_points_on_s3(num_points)

# Function to create geodesic lines
def generate_geodesic_lines(points, num_points_per_line=100):
    geodesic_lines = [[] for _ in range(6)]
    colors = [[] for _ in range(6)]
    
    for i, point in enumerate(points):
        x, y, z, w = point
        geodesics = [
            (x, y, 0, 0),
            (x, 0, z, 0),
            (x, 0, 0, w),
            (0, y, 0, w),
            (0, 0, z, w),
            (0, y, z, 0)
        ]
        
        color_map = ['r', 'b', 'g', 'c', 'm', 'y']
        
        for j, (gx, gy, gz, gw) in enumerate(geodesics):
            t = np.linspace(0, 2 * np.pi, num_points_per_line)
            cos_t = np.cos(t)
            sin_t = np.sin(t)
            line = point * cos_t[:, np.newaxis] + np.array([gx, gy, gz, gw]) * sin_t[:, np.newaxis]
            geodesic_lines[j].append(line)
            colors[j].append(color_map[j % len(color_map)])
    
    return geodesic_lines, colors

# Generate geodesic lines and colors
geodesic_lines, line_colors = generate_geodesic_lines(points)

# Colorize specific lines
for i in range(len(line_colors[1])):  # (x, 0, z, 0) lines
    line_colors[1][i] = 'b'
for i in range(len(line_colors[5])):  # (0, y, z, 0) lines
    line_colors[5][i] = 'r'

# Choose a random black great circle from the set of blue lines
random_index = np.random.randint(0, len(geodesic_lines[1]))
black_geodesic = geodesic_lines[1][random_index]
black_color = 'k'

# Choose a random very dark grey great circle from the set satisfying (0, y, 0, w)
random_index_dark_grey = np.random.randint(0, len(geodesic_lines[3]))
dark_grey_geodesic = geodesic_lines[3][random_index_dark_grey]
dark_grey_color = 'dimgray'

# Select 8 equally distant points on the grey circle
num_points_grey_circle = 8
t_grey_circle = np.linspace(0, 2 * np.pi, num_points_grey_circle, endpoint=False)
cos_t_grey = np.cos(t_grey_circle)
sin_t_grey = np.sin(t_grey_circle)
points_on_grey = [
    dark_grey_geodesic[int(len(dark_grey_geodesic) * i / num_points_grey_circle)]
    for i in range(num_points_grey_circle)
]

# Select a single point on the black circle
point_on_black = black_geodesic[0]

# Function for different 2D projection methods
def project_points(points, method='orthographic'):
    if method == 'orthographic':
        return points[:, :2]
    elif method == 'stereographic':
        denominator = 1 - points[:, 3]
        projected_points = points[:, :2] / denominator[:, None]
        return projected_points
    elif method == 'azimuthal':
        x, y, z, w = points.T
        projected_points = np.vstack((np.arctan2(y, x), np.arccos(z / np.sqrt(x**2 + y**2 + z**2)))).T
        return projected_points
    # Add more projection methods as needed
    else:
        raise ValueError("Unknown projection method")

# Define projection methods
projection_methods = ['orthographic', 'stereographic', 'azimuthal']

# Function to create geodesic arcs
def create_geodesic_arc(point1, point2, num_points=100):
    t = np.linspace(0, 1, num_points)
    arc = (1 - t)[:, np.newaxis] * point1 + t[:, np.newaxis] * point2
    arc /= np.linalg.norm(arc, axis=1)[:, np.newaxis]  # Normalize to lie on S^3
    return arc

# Generate geodesic arcs
geodesic_arcs = [create_geodesic_arc(p, point_on_black) for p in points_on_grey]

# Function to plot points and lines using different projections
def plot_projections(points, geodesic_lines, line_colors, methods, black_geodesic, black_color, dark_grey_geodesic, dark_grey_color, geodesic_arcs):
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    axs = axs.flatten()

    for i, method in enumerate(methods):
        # Project points
        projected_points = project_points(points, method)
        
        # Plot points
        axs[i].scatter(projected_points[:, 0], projected_points[:, 1], s=1, alpha=0.5)
        axs[i].set_title(f'{method} projection')
        
        # Project and plot lines
        for k in range(6):
            for j, line in enumerate(geodesic_lines[k]):
                projected_line = project_points(line, method)
                axs[i].plot(
                    projected_line[:, 0], projected_line[:, 1], 
                    line_colors[k][j], linewidth=0.5, alpha=0.5
                )
        
        # Project and plot the black geodesic
        projected_black_geodesic = project_points(black_geodesic, method)
        axs[i].plot(
            projected_black_geodesic[:, 0], projected_black_geodesic[:, 1], 
            black_color, linewidth=2, alpha=0.8
        )

        # Project and plot the dark grey geodesic
        projected_dark_grey_geodesic = project_points(dark_grey_geodesic, method)
        axs[i].plot(
            projected_dark_grey_geodesic[:, 0], projected_dark_grey_geodesic[:, 1], 
            dark_grey_color, linewidth=2, alpha=0.8
        )
        
        # Project and plot the geodesic arcs
        for arc in geodesic_arcs:
            projected_arc = project_points(arc, method)
            axs[i].plot(
                projected_arc[:, 0], projected_arc[:, 1], 
                'k--', linewidth=0.5, alpha=0.8
            )
        
        axs[i].set_xlim([-2, 2])
        axs[i].set_ylim([-2, 2])
    
    plt.tight_layout()
    plt.show()

# Plot the projections
plot_projections(points, geodesic_lines, line_colors, projection_methods, black_geodesic, black_color, dark_grey_geodesic, dark_grey_color, geodesic_arcs)
