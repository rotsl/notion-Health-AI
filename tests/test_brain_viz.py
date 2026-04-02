# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Rohan R
# Author: Rohan R

"""Tests for visualization timeline alignment helpers."""

import numpy as np

from notion_health_ai.brain_viz import BrainVisualizer


class TestBrainVisualizerTimeline:
    """Test visualization timeline alignment helpers."""

    def test_align_timesteps_matches_simulation_duration(self):
        """Multi-step predictions should be resampled to one frame per simulated minute."""
        preds = np.array(
            [
                [0.1, 0.2],
                [0.4, 0.5],
                [0.8, 0.9],
            ],
            dtype=np.float32,
        )

        aligned = BrainVisualizer._align_timesteps(preds, duration_minutes=6)

        assert aligned.shape == (6, 2)
        np.testing.assert_allclose(aligned[0], preds[0])
        np.testing.assert_allclose(aligned[-1], preds[-1])

    def test_time_labels_use_minutes_and_hours(self):
        """Timeline labels should reflect simulated minutes instead of generic seconds."""
        labels = BrainVisualizer._time_labels(4, duration_minutes=120)

        assert labels[0] == "1 min"
        assert labels[-1] == "2 hrs"
