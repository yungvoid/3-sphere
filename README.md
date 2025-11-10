# 3-sphere visualizations (S³) and lens-like animations

This repo contains small Python scripts to visualize structures in the lens space build from the 3‑sphere S³ and project them for plotting. You can generate static plots and animated GIFs of great circles, arc “skeletons,” and the effect of rotating S³ on an embedded 2‑sphere.

- `Lens.py` — produces a GIF of the effect of a rotation of S³ on a sphere embedded in S³. This embedded S² corresponds to a lens in the associated lens space; the animation makes this “lens” sweep/rotate under the S³ action.
- `S3.py` — helper experiments with S³ projections/plots.

Images and intermediate frames are written under `frames/` and to `.gif` files (both ignored by Git by default via `.gitignore`).

## Install and run

Use a Python 3.10+ virtual environment. On macOS with zsh:

```zsh
# from the project directory
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Quick start: create a GIF with `Lens.py`

`Lens.py` shows how a rotation of S³ acts on an embedded sphere (S²) in S³. In lens-space language, this S² corresponds to a “lens”; the animation demonstrates how that lens moves under the rotation.

```zsh
# Export a short GIF
python Lens.py --gif --gif-path lens.gif --fps 10 --steps 16
```

- `--gif` exports to a GIF instead of showing an interactive window.
- `--gif-path` chooses the output filename.
- `--steps` controls the number of frames (defaults to number of edge points + 1).
- `--fps` sets playback speed.
- `--no-surface` and `--no-interior` can hide parts of the arc scaffolding.
- `--interactive` forces the original interactive (plt.pause) animation instead of GIF export.

Tip: run `python Lens.py --help` for the current list of flags in your version.

## Mathematical notes (informal)

- S³ can be visualized through families of great circles and geodesic arcs. We parameterize great circles as linear combinations of two orthonormal vectors in R⁴ and project to R³ with stereographic projection.
- In `Lens.py`, we focus on an embedded 2‑sphere inside S³ and animate the effect of an S³ rotation. In the context of lens spaces (quotients of S³ by a finite cyclic isometry group), such embedded spheres correspond to “lenses.” The animation gives intuition for how a single lens moves under the ambient S³ rotation, though it does not itself perform the lens-space quotient.

## Troubleshooting

- If you get an error about missing `imageio`, install dependencies:

```zsh
pip install -r requirements.txt
```

- If plots look clipped, you can adjust the number of points or disable dense arc families with `--no-surface`/`--no-interior`.

- If your terminal is noisy, note that DEBUG logs are suppressed by default; only concise INFO lines are printed when exporting GIFs.

## Project structure (selected)

- `Lens.py` — rotation of S³ acting on an embedded S² (lens) with GIF export.
- `S3.py` — simple S³ helpers/experiments.
- `requirements.txt` — pinned Python dependencies (numpy, matplotlib, imageio).
- `frames/` — temporary PNG frames when exporting GIFs.

## License

No license specified. If you intend to publish or share, consider adding a LICENSE file.
