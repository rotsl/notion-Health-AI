"""
Brain activation visualization module for TRIBEv2 predictions.

Generates multiple visualization types from TRIBEv2 fMRI predictions:
  - Interactive 3D HTML (plotly)  — live, opens in browser, rotatable, time-scrubbing
  - Static 4-view PNG (nilearn)   — lateral + medial views for both hemispheres
  - Animated GIF (nilearn + imageio) — timestep-by-timestep brain activation
  - MP4 video (TRIBEv2 native or matplotlib) — smooth animation
  - ROI heatmap (matplotlib)      — always works, no brain mesh needed

All outputs are saved to <project_root>/visualizations/.
"""

from __future__ import annotations

import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np

from loguru import logger

# ── Optional dependency guards ──────────────────────────────────────────────

try:
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from nilearn import datasets as _nl_datasets, surface as _nl_surface

    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend; safe for server/CLI use
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import imageio.v2 as imageio

    IMAGEIO_AVAILABLE = True
except ImportError:
    try:
        import imageio

        IMAGEIO_AVAILABLE = True
    except ImportError:
        IMAGEIO_AVAILABLE = False

try:
    import pyvista as pv

    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False

# ── Project paths ────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve()
PROJECT_ROOT = _HERE.parent.parent.parent  # src/notion_health_ai → src → project root
VIZ_DIR = PROJECT_ROOT / "visualizations"

# Number of vertices per hemisphere in fsaverage5
_N_VERTS_PER_HEMI = 10242  # 10242 × 2 = 20484 total


# ── Mesh cache ───────────────────────────────────────────────────────────────


class _MeshCache:
    """Singleton: loads fsaverage5 mesh once and caches it."""

    _coords_lh: Optional[np.ndarray] = None
    _faces_lh: Optional[np.ndarray] = None
    _coords_rh: Optional[np.ndarray] = None
    _faces_rh: Optional[np.ndarray] = None
    _loaded: bool = False
    _surf_lh: Optional[str] = None  # nilearn surface file paths
    _surf_rh: Optional[str] = None

    @classmethod
    def load(cls):
        if cls._loaded:
            return
        if not NILEARN_AVAILABLE:
            raise ImportError(
                "nilearn is required for brain surface visualization.\n"
                "Install with: pip install nilearn nibabel"
            )
        logger.info("Loading fsaverage5 mesh (one-time download ~5 MB)...")
        fsaverage = _nl_datasets.fetch_surf_fsaverage("fsaverage5")
        cls._surf_lh = fsaverage.infl_left
        cls._surf_rh = fsaverage.infl_right
        cls._coords_lh, cls._faces_lh = _nl_surface.load_surf_mesh(fsaverage.infl_left)
        cls._coords_rh, cls._faces_rh = _nl_surface.load_surf_mesh(fsaverage.infl_right)
        cls._loaded = True
        logger.info("fsaverage5 mesh loaded.")


# ── Main visualizer class ────────────────────────────────────────────────────


class BrainVisualizer:
    """
    Visualize TRIBEv2 cortical activation predictions.

    Usage::

        viz = BrainVisualizer()

        # preds shape: (n_vertices,) or (n_timesteps, n_vertices)
        path = viz.plot_interactive(preds, title="Running 30 min", auto_open=True)
        path = viz.plot_static(preds.mean(0), title="Running 30 min")
        path = viz.plot_gif(preds, title="Video prediction")
        path = viz.plot_roi_heatmap(roi_dict, title="ROI activations")
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        self.output_dir = Path(output_dir or VIZ_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _ts(prefix: str) -> str:
        """Short timestamp suffix for unique filenames."""
        return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    @staticmethod
    def _split_hemispheres(preds_1d: np.ndarray):
        """Split flat (n_vertices,) array into LH and RH halves."""
        n = len(preds_1d)
        half = n // 2
        return preds_1d[:half], preds_1d[half:]

    @staticmethod
    def _ensure_1d(preds: np.ndarray) -> np.ndarray:
        """If 2-D (n_timesteps, n_vertices), return mean across time."""
        return preds.mean(axis=0) if preds.ndim == 2 else preds

    # ── Interactive 3D HTML (plotly) ──────────────────────────────────────

    def plot_interactive(
        self,
        preds: np.ndarray,
        title: str = "Brain Activation",
        output_name: Optional[str] = None,
        colorscale: str = "hot",
        auto_open: bool = True,
    ) -> Path:
        """
        Generate a live interactive 3D brain visualization as an HTML file.

        For multi-timestep predictions (shape: n_timesteps × n_vertices) a
        Play/Pause button and time-scrubber slider are added automatically.

        The file is auto-opened in the default browser if *auto_open* is True.

        Args:
            preds: ``(n_vertices,)`` or ``(n_timesteps, n_vertices)``
            title: Plot title shown in the browser tab and header
            output_name: Filename stem (no extension); auto-generated if None
            colorscale: Any plotly colorscale name (default "hot")
            auto_open: Open the generated HTML in the default browser

        Returns:
            Path to the saved .html file
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly is required: pip install plotly")
        _MeshCache.load()

        # Normalise to 2-D
        if preds.ndim == 1:
            preds_2d = preds[np.newaxis, :]
        else:
            preds_2d = preds
        n_timesteps, n_verts = preds_2d.shape

        coords_lh = _MeshCache._coords_lh
        faces_lh = _MeshCache._faces_lh
        coords_rh = _MeshCache._coords_rh.copy()
        faces_rh = _MeshCache._faces_rh
        # Offset RH so both hemispheres sit side-by-side
        x_gap = (coords_lh[:, 0].max() - coords_rh[:, 0].min()) + 25
        coords_rh[:, 0] += x_gap

        half = min(_N_VERTS_PER_HEMI, n_verts // 2)
        vmin = float(np.percentile(preds_2d, 2))
        vmax = float(np.percentile(preds_2d, 98))

        def _mesh_traces(t: int) -> List[go.Mesh3d]:
            d_lh = preds_2d[t, :half].tolist()
            d_rh = preds_2d[t, half : half * 2].tolist()
            base = dict(
                colorscale=colorscale,
                cmin=vmin,
                cmax=vmax,
                lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.5),
                lightposition=dict(x=100, y=200, z=150),
                hoverinfo="skip",
            )
            lh = go.Mesh3d(
                x=coords_lh[:, 0],
                y=coords_lh[:, 1],
                z=coords_lh[:, 2],
                i=faces_lh[:, 0],
                j=faces_lh[:, 1],
                k=faces_lh[:, 2],
                intensity=d_lh,
                showscale=False,
                name="Left",
                **base,
            )
            rh = go.Mesh3d(
                x=coords_rh[:, 0],
                y=coords_rh[:, 1],
                z=coords_rh[:, 2],
                i=faces_rh[:, 0],
                j=faces_rh[:, 1],
                k=faces_rh[:, 2],
                intensity=d_rh,
                showscale=True,
                name="Right",
                colorbar=dict(
                    title=dict(text="Activation", font=dict(color="white")),
                    tickfont=dict(color="white"),
                    thickness=18,
                    len=0.7,
                ),
                **base,
            )
            return [lh, rh]

        scene = dict(
            xaxis=dict(visible=False, showgrid=False),
            yaxis=dict(visible=False, showgrid=False),
            zaxis=dict(visible=False, showgrid=False),
            bgcolor="rgb(8, 8, 20)",
            camera=dict(eye=dict(x=0.0, y=-2.2, z=0.5)),
            aspectmode="data",
        )
        layout_base = dict(
            title=dict(text=title, x=0.5, font=dict(size=20, color="white")),
            paper_bgcolor="rgb(10, 10, 28)",
            font=dict(color="white"),
            margin=dict(l=0, r=0, t=55, b=55),
            height=680,
            scene=scene,
        )

        if n_timesteps == 1:
            fig = go.Figure(data=_mesh_traces(0), layout=go.Layout(**layout_base))
        else:
            # Build animation frames + slider + play/pause buttons
            frames = [go.Frame(data=_mesh_traces(t), name=str(t)) for t in range(n_timesteps)]
            slider_steps = [
                dict(
                    args=[[str(t)], dict(mode="immediate", frame=dict(duration=400, redraw=True))],
                    method="animate",
                    label=f"{t}s",
                )
                for t in range(n_timesteps)
            ]
            layout_base["updatemenus"] = [
                dict(
                    type="buttons",
                    showactive=False,
                    y=-0.08,
                    x=0.5,
                    xanchor="center",
                    buttons=[
                        dict(
                            label="▶  Play",
                            method="animate",
                            args=[
                                None,
                                dict(
                                    frame=dict(duration=450, redraw=True),
                                    fromcurrent=True,
                                    transition=dict(duration=0),
                                ),
                            ],
                        ),
                        dict(
                            label="⏸  Pause",
                            method="animate",
                            args=[
                                [None],
                                dict(mode="immediate", frame=dict(duration=0, redraw=False)),
                            ],
                        ),
                    ],
                    font=dict(color="black"),
                )
            ]
            layout_base["sliders"] = [
                dict(
                    active=0,
                    steps=slider_steps,
                    currentvalue=dict(prefix="Timestep: ", font=dict(color="white")),
                    font=dict(color="white"),
                    bgcolor="rgba(255,255,255,0.1)",
                    bordercolor="rgba(255,255,255,0.3)",
                    x=0.05,
                    len=0.9,
                    y=-0.02,
                )
            ]
            fig = go.Figure(
                data=_mesh_traces(0),
                frames=frames,
                layout=go.Layout(**layout_base),
            )

        # Add annotation: modality info
        if n_timesteps > 1:
            fig.add_annotation(
                text=f"TRIBEv2 · {n_timesteps} timesteps · {n_verts} cortical vertices",
                xref="paper",
                yref="paper",
                x=0.5,
                y=-0.14,
                showarrow=False,
                font=dict(size=11, color="rgba(255,255,255,0.5)"),
            )

        out_name = output_name or self._ts("brain_interactive")
        out_path = self.output_dir / f"{out_name}.html"
        fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
        logger.info(f"Interactive visualization saved: {out_path}")

        if auto_open:
            webbrowser.open(f"file://{out_path.resolve()}")

        return out_path

    # ── Static 4-view PNG (nilearn) ───────────────────────────────────────

    def plot_static(
        self,
        preds: np.ndarray,
        title: str = "Brain Activation",
        output_name: Optional[str] = None,
        cmap: str = "hot",
    ) -> Path:
        """
        Generate a static 4-panel PNG (lateral-L, medial-L, lateral-R, medial-R).

        Args:
            preds: ``(n_vertices,)`` — averaged across time if 2-D is passed
            title: Figure title
            output_name: Filename stem
            cmap: matplotlib colormap name

        Returns:
            Path to the saved .png file
        """
        if not NILEARN_AVAILABLE:
            raise ImportError("nilearn + nibabel required: pip install nilearn nibabel")
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required: pip install matplotlib")

        from nilearn import datasets, plotting

        preds_1d = self._ensure_1d(preds)
        half = min(_N_VERTS_PER_HEMI, len(preds_1d) // 2)
        d_lh = preds_1d[:half]
        d_rh = preds_1d[half : half * 2]
        vmax = float(np.percentile(preds_1d, 99))

        fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
        panels = [
            (fsaverage.infl_left, d_lh, "left", "lateral", "Lateral Left"),
            (fsaverage.infl_left, d_lh, "left", "medial", "Medial Left"),
            (fsaverage.infl_right, d_rh, "right", "lateral", "Lateral Right"),
            (fsaverage.infl_right, d_rh, "right", "medial", "Medial Right"),
        ]

        fig, axes = plt.subplots(1, 4, figsize=(22, 5), subplot_kw={"projection": "3d"})
        fig.suptitle(title, fontsize=14, y=1.0, weight="bold")

        for ax, (mesh, data, hemi, view, label) in zip(axes, panels):
            try:
                plotting.plot_surf_stat_map(
                    mesh,
                    data,
                    hemi=hemi,
                    view=view,
                    cmap=cmap,
                    vmax=vmax,
                    colorbar=False,
                    bg_on_data=True,
                    axes=ax,
                )
            except TypeError:
                # Older nilearn: no axes kwarg — render separately
                disp = plotting.plot_surf_stat_map(
                    mesh,
                    data,
                    hemi=hemi,
                    view=view,
                    cmap=cmap,
                    vmax=vmax,
                    colorbar=False,
                    bg_on_data=True,
                )
                disp.axes.set_title(label)
                disp.savefig(
                    str(self.output_dir / f"_panel_{hemi}_{view}.png"),
                    dpi=100,
                )
                disp.close()
                ax.set_visible(False)
                continue
            ax.set_title(label, fontsize=10, pad=4)

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        out_name = output_name or self._ts("brain_static")
        out_path = self.output_dir / f"{out_name}.png"
        plt.savefig(
            str(out_path), dpi=150, bbox_inches="tight", facecolor="white", transparent=False
        )
        plt.close(fig)
        logger.info(f"Static visualization saved: {out_path}")
        return out_path

    # ── Animated GIF ──────────────────────────────────────────────────────

    def plot_gif(
        self,
        preds: np.ndarray,
        title: str = "Brain Activation Over Time",
        output_name: Optional[str] = None,
        fps: int = 2,
        max_frames: int = 30,
        cmap: str = "hot",
    ) -> Path:
        """
        Animate brain activation across timesteps as a GIF.

        Args:
            preds: ``(n_timesteps, n_vertices)``
            title: Animation title
            output_name: Filename stem
            fps: Frames per second
            max_frames: Maximum number of timesteps to include
            cmap: matplotlib colormap

        Returns:
            Path to the saved .gif file
        """
        if not NILEARN_AVAILABLE:
            raise ImportError("nilearn + nibabel required: pip install nilearn nibabel")
        if not IMAGEIO_AVAILABLE:
            raise ImportError("imageio required: pip install imageio")

        from nilearn import datasets, plotting

        if preds.ndim == 1:
            preds = preds[np.newaxis, :]

        fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
        half = min(_N_VERTS_PER_HEMI, preds.shape[1] // 2)
        n_frames = min(preds.shape[0], max_frames)
        vmax = float(np.percentile(preds, 98))

        tmp_dir = self.output_dir / "_tmp_gif_frames"
        tmp_dir.mkdir(exist_ok=True)
        frame_paths: List[Path] = []

        for t in range(n_frames):
            fig, axes = plt.subplots(1, 2, figsize=(14, 4), subplot_kw={"projection": "3d"})
            fig.suptitle(f"{title}  [t = {t}s]", fontsize=11)

            for ax, (mesh, data, hemi, label) in zip(
                axes,
                [
                    (fsaverage.infl_left, preds[t, :half], "left", "Left"),
                    (fsaverage.infl_right, preds[t, half : half * 2], "right", "Right"),
                ],
            ):
                try:
                    plotting.plot_surf_stat_map(
                        mesh,
                        data,
                        hemi=hemi,
                        view="lateral",
                        cmap=cmap,
                        vmax=vmax,
                        colorbar=False,
                        bg_on_data=True,
                        axes=ax,
                    )
                except TypeError:
                    plotting.plot_surf_stat_map(
                        mesh,
                        data,
                        hemi=hemi,
                        view="lateral",
                        cmap=cmap,
                        vmax=vmax,
                        colorbar=False,
                        bg_on_data=True,
                    )
                ax.set_title(label, fontsize=9)

            fp = tmp_dir / f"f{t:04d}.png"
            plt.savefig(str(fp), dpi=90, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            frame_paths.append(fp)

        out_name = output_name or self._ts("brain_animated")
        out_path = self.output_dir / f"{out_name}.gif"
        images = [imageio.imread(str(fp)) for fp in frame_paths]
        imageio.mimsave(str(out_path), images, fps=fps, loop=0)

        for fp in frame_paths:
            fp.unlink(missing_ok=True)
        try:
            tmp_dir.rmdir()
        except OSError:
            pass

        logger.info(f"Animated GIF saved: {out_path}")
        return out_path

    # ── MP4 video ─────────────────────────────────────────────────────────

    def plot_mp4(
        self,
        preds: np.ndarray,
        title: str = "Brain Activation Over Time",
        output_name: Optional[str] = None,
        fps: int = 2,
        max_frames: int = 60,
    ) -> Path:
        """
        Export brain activation over time as an MP4 video.

        Tries TRIBEv2's built-in ``PlotBrain.plot_timesteps_mp4`` first
        (requires pyvista); falls back to matplotlib + ffmpeg.

        Args:
            preds: ``(n_timesteps, n_vertices)``
            title: Video title
            output_name: Filename stem
            fps: Frames per second
            max_frames: Max timesteps to include

        Returns:
            Path to the saved .mp4 file
        """
        if preds.ndim == 1:
            preds = preds[np.newaxis, :]

        out_name = output_name or self._ts("brain_video")
        out_path = self.output_dir / f"{out_name}.mp4"
        n_frames = min(preds.shape[0], max_frames)

        # ── Try TRIBEv2 native plotter ───────────────────────────────────
        try:
            from tribev2.plotting import PlotBrain

            plotter = PlotBrain(mesh="fsaverage5")
            plotter.plot_timesteps_mp4(
                preds,
                timesteps=list(range(n_frames)),
                output_path=str(out_path),
                fps=fps,
            )
            logger.info(f"MP4 (TRIBEv2 native) saved: {out_path}")
            return out_path
        except Exception as e:
            logger.debug(f"TRIBEv2 native MP4 failed ({e}), falling back to matplotlib")

        # ── matplotlib + ffmpeg fallback ─────────────────────────────────
        if not NILEARN_AVAILABLE:
            raise ImportError("nilearn required for MP4 fallback: pip install nilearn nibabel")
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for MP4 fallback: pip install matplotlib")

        from nilearn import datasets, plotting

        fsaverage = datasets.fetch_surf_fsaverage("fsaverage5")
        half = min(_N_VERTS_PER_HEMI, preds.shape[1] // 2)
        vmax = float(np.percentile(preds, 98))

        fig, axes = plt.subplots(1, 2, figsize=(14, 4), subplot_kw={"projection": "3d"})

        def _render_frame(t: int):
            for ax in axes:
                ax.clear()
            for ax, (mesh, data, hemi, lbl) in zip(
                axes,
                [
                    (fsaverage.infl_left, preds[t, :half], "left", "LH"),
                    (fsaverage.infl_right, preds[t, half : half * 2], "right", "RH"),
                ],
            ):
                try:
                    plotting.plot_surf_stat_map(
                        mesh,
                        data,
                        hemi=hemi,
                        view="lateral",
                        cmap="hot",
                        vmax=vmax,
                        colorbar=False,
                        bg_on_data=True,
                        axes=ax,
                    )
                except TypeError:
                    pass
                ax.set_title(f"{lbl}  t={t}s", fontsize=9)
            fig.suptitle(title, fontsize=11)

        anim = animation.FuncAnimation(
            fig,
            _render_frame,
            frames=n_frames,
            interval=int(1000 / fps),
        )
        anim.save(str(out_path), writer="ffmpeg", fps=fps, dpi=100)
        plt.close(fig)
        logger.info(f"MP4 (matplotlib) saved: {out_path}")
        return out_path

    # ── ROI heatmap (no brain mesh needed) ────────────────────────────────

    def plot_roi_heatmap(
        self,
        roi_activations: Dict[str, float],
        title: str = "Brain Region Activations",
        output_name: Optional[str] = None,
        modality: str = "",
    ) -> Path:
        """
        Horizontal bar-chart of ROI activation values.

        Works with matplotlib only — no nilearn or plotly required.
        Always generated as a companion to full surface plots.

        Args:
            roi_activations: ``{region_name: activation_value (0–1)}``
            title: Chart title
            output_name: Filename stem
            modality: Label shown in subtitle (e.g. "video", "audio", "text")

        Returns:
            Path to the saved .png file
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required: pip install matplotlib")

        regions = list(roi_activations.keys())
        values = [float(roi_activations[r]) for r in regions]
        cmap = plt.cm.hot
        colours = cmap(np.clip(values, 0, 1))

        fig, ax = plt.subplots(figsize=(10, max(4, len(regions) * 0.55)))
        bars = ax.barh(regions, values, color=colours, edgecolor="none", height=0.65)
        ax.set_xlim(0, 1.1)
        ax.set_xlabel("Activation Level (0–1)", fontsize=11)
        subtitle = f"  [{modality}]" if modality else ""
        ax.set_title(f"{title}{subtitle}", fontsize=13, weight="bold", pad=10)
        ax.spines[["top", "right"]].set_visible(False)

        # Value labels inside bars
        for bar, val in zip(bars, values):
            x_pos = val + 0.02 if val < 0.9 else val - 0.06
            ax.text(
                x_pos,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}",
                va="center",
                ha="left",
                fontsize=9,
                color="black" if val < 0.9 else "white",
            )

        # Activation-level legend lines
        for lvl, lbl, col in [(0.25, "threshold", "steelblue"), (0.4, "excitatory", "tomato")]:
            ax.axvline(lvl, color=col, linestyle="--", linewidth=1, alpha=0.6)
            ax.text(lvl + 0.01, len(regions) - 0.4, lbl, fontsize=8, color=col, alpha=0.8)

        plt.tight_layout()
        out_name = output_name or self._ts("roi_heatmap")
        out_path = self.output_dir / f"{out_name}.png"
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        logger.info(f"ROI heatmap saved: {out_path}")
        return out_path

    # ── Live PyVista window ───────────────────────────────────────────────

    def plot_live_pyvista(
        self,
        preds: np.ndarray,
        title: str = "Brain Activation (Live)",
        timestep: int = 0,
    ) -> None:
        """
        Open an interactive PyVista 3D window (requires pyvista + display server).

        Rotatable in real-time; useful during local development.
        Falls back gracefully to a log message if pyvista is unavailable.

        Args:
            preds: ``(n_vertices,)`` or ``(n_timesteps, n_vertices)``
            title: Window title
            timestep: Which timestep to show (0 = mean if 1-D)
        """
        if not PYVISTA_AVAILABLE:
            logger.warning(
                "pyvista not available — install with: pip install pyvista\n"
                "Falling back to interactive HTML visualization."
            )
            self.plot_interactive(preds, title=title, auto_open=True)
            return

        try:
            from tribev2.plotting import PlotBrain

            plotter = PlotBrain(mesh="fsaverage5")
            if preds.ndim == 2:
                data = preds[timestep]
            else:
                data = preds
            plotter.plot_surf(data, cmap="fire", title=title)
        except Exception as e:
            logger.warning(f"TRIBEv2 PlotBrain failed ({e}), using plain PyVista")
            _MeshCache.load()
            data_1d = self._ensure_1d(preds)
            half = min(_N_VERTS_PER_HEMI, len(data_1d) // 2)

            mesh_l = pv.PolyData(
                _MeshCache._coords_lh,
                np.hstack(
                    [np.full((len(_MeshCache._faces_lh), 1), 3), _MeshCache._faces_lh]
                ).astype(np.int64),
            )
            mesh_r = pv.PolyData(
                _MeshCache._coords_rh,
                np.hstack(
                    [np.full((len(_MeshCache._faces_rh), 1), 3), _MeshCache._faces_rh]
                ).astype(np.int64),
            )
            mesh_l["activation"] = data_1d[:half]
            mesh_r["activation"] = data_1d[half : half * 2]

            pl = pv.Plotter(title=title)
            pl.add_mesh(mesh_l, scalars="activation", cmap="hot", show_scalar_bar=False)
            pl.add_mesh(
                mesh_r, scalars="activation", cmap="hot", scalar_bar_args={"title": "Activation"}
            )
            pl.add_text(title, font_size=12)
            pl.show()

    # ── Convenience: generate all standard visualizations ─────────────────

    def generate_all(
        self,
        preds: np.ndarray,
        roi_activations: Dict[str, float],
        title: str = "Brain Activation",
        output_stem: str = "brain",
        viz_types: Optional[List[str]] = None,
        auto_open_interactive: bool = True,
        modality: str = "",
    ) -> Dict[str, Optional[str]]:
        """
        Generate all requested visualization types for a prediction.

        Args:
            preds: Raw TRIBEv2 predictions ``(n_timesteps, n_vertices)``
            roi_activations: ``{region: activation}`` dict from ``_process_tribe_outputs``
            title: Shared title for all plots
            output_stem: Filename prefix (timestamp appended)
            viz_types: Subset of ["interactive", "static", "gif", "mp4", "heatmap"]
                       Defaults to ["interactive", "heatmap"]
            auto_open_interactive: Auto-open HTML in browser
            modality: Label for ROI heatmap subtitle

        Returns:
            Dict mapping viz type → saved file path (str) or None on failure
        """
        viz_types = viz_types or ["interactive", "heatmap"]
        stem = f"{output_stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        results: Dict[str, Optional[str]] = {}

        for vtype in viz_types:
            try:
                if vtype == "interactive":
                    p = self.plot_interactive(
                        preds,
                        title=title,
                        output_name=f"{stem}_interactive",
                        auto_open=auto_open_interactive,
                    )
                    results["interactive"] = str(p)

                elif vtype == "static":
                    p = self.plot_static(
                        preds,
                        title=title,
                        output_name=f"{stem}_static",
                    )
                    results["static"] = str(p)

                elif vtype == "gif":
                    if preds.ndim == 2 and preds.shape[0] > 1:
                        p = self.plot_gif(preds, title=title, output_name=f"{stem}_anim")
                        results["gif"] = str(p)
                    else:
                        logger.info("Skipping GIF — single timestep prediction")

                elif vtype == "mp4":
                    if preds.ndim == 2 and preds.shape[0] > 1:
                        p = self.plot_mp4(preds, title=title, output_name=f"{stem}_video")
                        results["mp4"] = str(p)
                    else:
                        logger.info("Skipping MP4 — single timestep prediction")

                elif vtype == "heatmap":
                    p = self.plot_roi_heatmap(
                        roi_activations,
                        title=title,
                        output_name=f"{stem}_heatmap",
                        modality=modality,
                    )
                    results["heatmap"] = str(p)

                else:
                    logger.warning(f"Unknown viz type: {vtype!r}")

            except Exception as e:
                logger.warning(f"Visualization [{vtype}] failed: {e}")
                results[vtype] = None

        return results


# ── Module-level convenience wrapper ─────────────────────────────────────────

_default_viz: Optional[BrainVisualizer] = None


def get_visualizer() -> BrainVisualizer:
    """Return (or create) the module-level default BrainVisualizer."""
    global _default_viz
    if _default_viz is None:
        _default_viz = BrainVisualizer()
    return _default_viz


def open_file(path: Union[str, Path]) -> None:
    """Open any file in the default OS application (browser for HTML, Preview for PNG)."""
    webbrowser.open(f"file://{Path(path).resolve()}")
