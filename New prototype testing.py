import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- Parameters ---
L = 200  # beam width (y-direction)
theta_polarizer = np.radians(45)  # polarizer angle (45°)
theta_reflect = np.radians(30)     # mirror reflection angle (30° from normal)

# Ray directions (unit vectors)
rays = {
    'H': np.array([0.0, np.sin(theta_polarizer)]),  # |H〉 polarized ray
    'V': np.array([0.0, np.cos(theta_polarizer)])   # |V〉 polarized ray
}

# --- Setup figure ---
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_xlim(-20, 20)
ax.set_ylim(0, L + 10)
ax.set_aspect('equal')
ax.grid(True, linestyle='--', linewidth=0.5)
ax.set_title("2D Ray Optics Simulation: Polarization & Reflection", fontsize=14)

# --- Background grid ---
ax.add_patch(Rectangle((-10, 0), 20, L, facecolor='white', alpha=0.2))

# --- Draw laser source ---
laser_pos = (0, 0)
ax.plot(laser_pos[0], laser_pos[1], 'ro', markersize=8, label='Laser (Origin)')
ax.text(laser_pos[0], laser_pos[1] - 0.15, r'$\vec{E}_0$', fontsize=12, ha='center')

# --- Polarizer at 45° ---
pol_polarizer_pos = (0, L/2)
ax.plot(pol_polarizer_pos[0], pol_polarizer_pos[1], 'go', markersize=6, label='Polarizer (45°)')
ax.text(pol_polarizer_pos[0], pol_polarizer_pos[1] - 0.15, r'45°', fontsize=12, ha='center')

# --- Ray paths and polarization vectors ---
colors = {'H': 'b', 'V': 'r'}
for label, ray_dir in rays.items():
    # Ray path: horizontal line from origin to y=L
    y = L
    ray_line, = ax.plot([0, L], [0, y], 'k-', lw=2, label=f'{label} Ray')
    
    # Polarization vector at start (H or V)
    pvec = ray_dir * np.array([1, label == 'V'])  # |H⟩ = (1,0), |V⟩ = (0,1)
    p_len = 0.05  # scale for visibility
    p_arrow, = ax.plot(
        0, 0, pvec + np.array([0, p_len]), 'yo', markersize=p_len*2, label=f'{label} Pol'
    )
    ax.text(0, pvec[1] + 0.1, f'{label} Pol', fontsize=10, ha='center', color='black')

# --- Beam splitter (50:50) at center y=L/2 ---
bs_pos = (0, L/2)
ax.plot(bs_pos[0], bs_pos[1], 'go', markersize=8, label='50:50 NPBS')
ax.text(bs_pos[0], bs_pos[1] - 0.15, r'50:50 NPBS', fontsize=12, ha='center')

# --- Mirrors at 30° reflection angle ---
mirror_pos = (L/2, L/2)
mirror_angle = -theta_reflect  # negative because mirror normal points upward

# Two rays: one incident, one reflected
rays_reflected = []

# Incident ray: horizontal toward mirror normal
incident_ray = np.array([1, np.tan(np.deg2rad(theta_reflect))])  # angle theta_reflect from normal
incident_ray /= np.linalg.norm(incident_ray)

# Reflected ray direction (angle = incidence = reflection)
reflected_ray = incident_ray * np.cos(2 * mirror_angle) + \
                np.array([np.sin(2 * mirror_angle), -np.sin(np.deg2rad(theta_reflect))]) * np.sin(2 * mirror_angle)

# Draw incident and reflected rays
for ray in [incident_ray, reflected_ray]:
    y = L
    ray_line, = ax.plot([0, L], [0, y], 'k-', lw=2)
    ax.text(0, y, r'→', fontsize=12, ha='center')

# Draw mirror line and normal
ax.plot([mirror_pos[0], mirror_pos[0]], [0, L], 'k--', lw=1.5, label='Mirror Normal')
ax.annotate('Normal', xy=(mirror_pos[0], L), xytext=(mirror_pos[0]+0.3, L+0.3),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=10)

# --- Detectors at bottom ---
detector_pos = (L, 0)
ax.plot(detector_pos[0], detector_pos[1], 'mo', markersize=12, label='Detector')
ax.text(detector_pos[0], detector_pos[1] - 0.15, r'Detector', fontsize=12, ha='center')

# --- Annotate polarization vectors on detector ---
# Assume detector collects |H⟩ and |V⟩ projections; draw arrows at detector with polarization info
polarization_vectors = {
    'H': np.array([1, 0]),
    'V': np.array([0, 1])
}

for label, pvec in polarization_vectors.items():
    # Scale and place polarization vectors near detector
    v_arrow, = ax.plot(
        detector_pos[0], detector_pos[1] - 0.25,
        pvec + np.array([0.3, 0]), linewidth=2
    )
    ax.text(detector_pos[0], detector_pos[1] - 0.28, f'{label} Pol', fontsize=10, ha='center', color='black')

# --- Final touches ---
ax.legend(loc='upper left', fontsize=10)
ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (Laser height, m)', fontsize=12)
ax.set_title('2D Ray Optics Simulation: Polarization Evolution and Reflection', fontsize=14, pad=20)

plt.tight_layout()
plt.show()