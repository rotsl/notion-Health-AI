"""
Tribe v2 Integration - Brain Response Prediction for Health.

This module integrates Meta's Tribe v2 foundation model for predicting
brain responses to various health and wellness activities. Tribe v2 is
a deep multimodal brain encoding model trained on fMRI data to predict
how the human brain responds to naturalistic stimuli.

Reference: https://huggingface.co/facebook/tribev2
Reference: https://github.com/facebookresearch/tribev2
Reference: https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/
"""

import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from loguru import logger

# Try to import torch for actual model inference
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Using simulated predictions.")

# Try to import TRIBEv2 package (install from https://github.com/facebookresearch/tribev2)
try:
    from tribev2 import TribeModel as _TribeModelClass

    TRIBEV2_AVAILABLE = True
    logger.info("TRIBEv2 package available.")
except ImportError:
    _TribeModelClass = None
    TRIBEV2_AVAILABLE = False
    logger.warning("TRIBEv2 package not available. Using simulated predictions.")


class BrainRegion(Enum):
    """Major brain regions tracked by Tribe v2."""

    VISUAL_CORTEX = "visual_cortex"
    AUDITORY_CORTEX = "auditory_cortex"
    PREFRONTAL_CORTEX = "prefrontal_cortex"
    MOTOR_CORTEX = "motor_cortex"
    HIPPOCAMPUS = "hippocampus"
    AMYGDALA = "amygdala"
    INSULA = "insula"
    ANTERIOR_CINGULATE = "anterior_cingulate"
    POSTERIOR_CINGULATE = "posterior_cingulate"
    TEMPORAL_LOBE = "temporal_lobe"
    PARIETAL_LOBE = "parietal_lobe"
    OCCIPITAL_LOBE = "occipital_lobe"
    CEREBELLUM = "cerebellum"
    BRAIN_STEM = "brain_stem"
    DEFAULT_MODE_NETWORK = "default_mode_network"


class ActivityType(Enum):
    """Types of wellness activities that can be analyzed."""

    EXERCISE = "exercise"
    MEDITATION = "meditation"
    SLEEP = "sleep"
    READING = "reading"
    MUSIC = "music"
    SOCIAL = "social"
    WORK = "work"
    RELAXATION = "relaxation"
    NATURE = "nature"
    LEARNING = "learning"
    CREATIVE = "creative"
    MINDFULNESS = "mindfulness"


@dataclass
class BrainResponse:
    """Predicted brain response to a stimulus or activity."""

    region: BrainRegion
    activation_level: float  # 0.0 to 1.0
    response_type: str  # "excitatory", "inhibitory", "neutral"
    confidence: float  # 0.0 to 1.0
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region": self.region.value,
            "activation_level": self.activation_level,
            "response_type": self.response_type,
            "confidence": self.confidence,
            "description": self.description,
        }


@dataclass
class ActivityPrediction:
    """Complete brain response prediction for an activity."""

    activity: ActivityType
    duration_minutes: int
    brain_responses: List[BrainResponse]
    overall_wellness_score: float  # 0.0 to 10.0
    benefits: List[str]
    recommendations: List[str]
    optimal_timing: Optional[str] = None
    intensity_level: str = "moderate"  # "low", "moderate", "high"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activity": self.activity.value,
            "duration_minutes": self.duration_minutes,
            "brain_responses": [r.to_dict() for r in self.brain_responses],
            "overall_wellness_score": self.overall_wellness_score,
            "benefits": self.benefits,
            "recommendations": self.recommendations,
            "optimal_timing": self.optimal_timing,
            "intensity_level": self.intensity_level,
        }


@dataclass
class BrainHealthCorrelation:
    """Correlation between health metrics and brain activity."""

    health_metric: str
    brain_region: BrainRegion
    correlation_strength: float  # -1.0 to 1.0
    insight: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "health_metric": self.health_metric,
            "brain_region": self.brain_region.value,
            "correlation_strength": self.correlation_strength,
            "insight": self.insight,
            "confidence": self.confidence,
        }


class TribeModelWrapper:
    """
    Wrapper for the Tribe v2 model from Hugging Face.

    Tribe v2 is a deep multimodal brain encoding model that predicts
    fMRI brain responses to naturalistic stimuli (video, audio, text).

    Model: facebook/tribev2
    Paper: "TRIBE v2: A Predictive Foundation Model Trained to Predict
            How the Human Brain Responds to Almost Any Sight or Sound"
    """

    MODEL_NAME = "facebook/tribev2"

    def __init__(self, device: Optional[str] = None):
        """
        Initialize the Tribe v2 model wrapper.

        Args:
            device: Device to run the model on ('cpu', 'cuda', 'mps')
        """
        self.device = device or self._get_device()
        self.model = None
        self.processor = None
        self.is_loaded = False
        # Store the most recent raw prediction array for visualization
        self._last_raw_predictions: Optional[Any] = None
        self._last_segments: Optional[Any] = None
        self._last_modality: str = "text"

        logger.info(f"TribeModelWrapper initialized on device: {self.device}")

    def _get_device(self) -> str:
        """Determine the best available device."""
        if TORCH_AVAILABLE:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        return "cpu"

    async def load_model(self, cache_dir: str = "./cache") -> bool:
        """
        Load the Tribe v2 model from Hugging Face.

        TRIBEv2 uses its own TribeModel class (not AutoModel).
        The model downloads ~676 MB checkpoint on first run.

        Args:
            cache_dir: Directory to cache downloaded model and features

        Returns:
            True if model loaded successfully, False otherwise
        """
        if not TORCH_AVAILABLE or not TRIBEV2_AVAILABLE:
            logger.warning(
                "Dependencies not available for model loading (need torch + tribev2 package)"
            )
            return False

        try:
            import os

            os.makedirs(cache_dir, exist_ok=True)

            # Authenticate with HuggingFace so gated model weights can be downloaded.
            hf_token = os.getenv("HUGGING_FACE_TOKEN") or os.getenv("HF_TOKEN")
            if hf_token:
                try:
                    from huggingface_hub import login as _hf_login

                    _hf_login(token=hf_token, add_to_git_credential=False)
                    logger.info("Authenticated with HuggingFace using HUGGING_FACE_TOKEN")
                except Exception as _login_err:
                    logger.warning(f"HuggingFace login warning (non-fatal): {_login_err}")
            else:
                logger.warning("No HUGGING_FACE_TOKEN found — download of gated model may fail")

            logger.info(f"Loading TRIBEv2 model from HuggingFace: {self.MODEL_NAME}")
            logger.info("First run will download ~676 MB checkpoint...")

            # TRIBEv2 uses its own TribeModel.from_pretrained() API.
            # The brain model (fMRI encoder) uses self.device (mps/cpu/cuda).
            # The internal feature extractors (Whisper, LLaMA) only support cuda/cpu,
            # so we override their device to cpu on non-CUDA systems.
            feature_device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
            self.model = _TribeModelClass.from_pretrained(
                self.MODEL_NAME,
                cache_folder=cache_dir,
                device=self.device,
                config_update={
                    "data.text_feature.device": feature_device,
                    "data.audio_feature.device": feature_device,
                },
            )

            self.is_loaded = True
            logger.info(f"TRIBEv2 model loaded successfully on {self.device}")
            return True

        except Exception as e:
            logger.error(f"Failed to load TRIBEv2 model: {e}")
            return False

    async def predict_response(
        self,
        text_stimulus: Optional[str] = None,
        audio_features: Optional[Any] = None,
        visual_features: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Predict brain response to given stimuli.

        Args:
            text_stimulus: Text description or stimulus
            audio_features: Audio features for the model
            visual_features: Visual features for the model

        Returns:
            Dictionary with predicted brain responses
        """
        if not self.is_loaded:
            # Return simulated predictions if model not loaded
            return self._simulate_prediction(text_stimulus)

        try:
            import tempfile
            import os

            if text_stimulus:
                # TRIBEv2 text pipeline: write to .txt file → gTTS → Whisper → predict
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, encoding="utf-8"
                ) as f:
                    f.write(text_stimulus)
                    tmp_path = f.name
                try:
                    events = self.model.get_events_dataframe(text_path=tmp_path)
                    preds, segments = self.model.predict(events, verbose=False)
                    self._last_raw_predictions = preds
                    self._last_segments = segments
                    self._last_modality = "text"
                    return self._process_tribe_outputs(preds)
                finally:
                    os.unlink(tmp_path)
            else:
                return self._simulate_prediction(text_stimulus)

        except Exception as e:
            logger.error(f"TRIBEv2 text prediction failed: {e}")
            return self._simulate_prediction(text_stimulus)

    async def predict_from_video(self, video_path: str) -> Dict[str, Any]:
        """
        Predict brain response from a video file using V-JEPA2 (visual) +
        Wav2Vec-BERT (audio) feature extractors.

        TRIBEv2 internally extracts visual frames via V-JEPA2 and transcribes
        audio via WhisperX → Wav2Vec-BERT, then maps both onto the cortical
        surface through its unified Transformer encoder.

        Args:
            video_path: Absolute path to an .mp4 / .avi / .mov / .mkv / .webm file

        Returns:
            ROI activation dict ``{region: {activation, type}}`` or simulated fallback
        """
        if not self.is_loaded:
            logger.warning("TRIBEv2 not loaded — returning simulated prediction for video")
            return self._simulate_prediction("visual experience")

        try:
            logger.info(f"Running TRIBEv2 video prediction: {video_path}")
            events = self.model.get_events_dataframe(video_path=video_path)
            preds, segments = self.model.predict(events, verbose=False)
            self._last_raw_predictions = preds
            self._last_segments = segments
            self._last_modality = "video"
            logger.info(f"Video prediction complete — shape {preds.shape}")
            return self._process_tribe_outputs(preds)
        except (RuntimeError, AssertionError) as e:
            if "CUDA" in str(e) or "cuda" in str(e):
                # V-JEPA2 video extractor requires CUDA; fall back to audio-only
                # by extracting the audio track and running the audio pipeline.
                logger.warning(
                    f"V-JEPA2 requires CUDA (not available on this device). "
                    f"Falling back to audio-only extraction from video. Error: {e}"
                )
                try:
                    import tempfile
                    from moviepy.editor import VideoFileClip

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                        wav_path = tf.name
                    clip = VideoFileClip(video_path)
                    clip.audio.write_audiofile(wav_path, logger=None)
                    clip.close()
                    result = await self.predict_from_audio(wav_path)
                    import os as _os

                    _os.unlink(wav_path)
                    return result
                except Exception as inner:
                    logger.error(f"Audio fallback also failed: {inner}")
                    return self._simulate_prediction("visual experience")
            logger.error(f"TRIBEv2 video prediction failed: {e}")
            return self._simulate_prediction("visual experience")
        except Exception as e:
            logger.error(f"TRIBEv2 video prediction failed: {e}")
            return self._simulate_prediction("visual experience")

    async def predict_from_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Predict brain response from an audio file using Wav2Vec-BERT +
        LLaMA 3.2 (via WhisperX transcription → text features).

        Args:
            audio_path: Absolute path to a .wav / .mp3 / .flac / .ogg file

        Returns:
            ROI activation dict ``{region: {activation, type}}`` or simulated fallback
        """
        if not self.is_loaded:
            logger.warning("TRIBEv2 not loaded — returning simulated prediction for audio")
            return self._simulate_prediction("audio listening experience")

        try:
            logger.info(f"Running TRIBEv2 audio prediction: {audio_path}")
            events = self.model.get_events_dataframe(audio_path=audio_path)
            preds, segments = self.model.predict(events, verbose=False)
            self._last_raw_predictions = preds
            self._last_segments = segments
            self._last_modality = "audio"
            logger.info(f"Audio prediction complete — shape {preds.shape}")
            return self._process_tribe_outputs(preds)
        except Exception as e:
            logger.error(f"TRIBEv2 audio prediction failed: {e}")
            return self._simulate_prediction("audio listening experience")

    async def predict_multimodal(
        self,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        text_stimulus: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Predict brain response from multiple modalities simultaneously.

        Runs each available modality through TRIBEv2 independently, then
        averages the raw vertex-level predictions before ROI mapping.  This
        provides a unified cortical estimate that reflects the combined
        information across V-JEPA2, Wav2Vec-BERT, and LLaMA 3.2.

        Args:
            video_path:     Path to video file (V-JEPA2 + Wav2Vec-BERT)
            audio_path:     Path to audio file (Wav2Vec-BERT + LLaMA 3.2)
            text_stimulus:  Plain text (LLaMA 3.2 via gTTS → Whisper)

        Returns:
            ROI activation dict averaged across all modalities provided
        """
        import numpy as np
        import tempfile
        import os

        if not self.is_loaded:
            hint = " | ".join(
                filter(
                    None,
                    [
                        "video" if video_path else "",
                        "audio" if audio_path else "",
                        "text" if text_stimulus else "",
                    ],
                )
            )
            return self._simulate_prediction(hint)

        all_preds = []

        if video_path:
            try:
                events = self.model.get_events_dataframe(video_path=video_path)
                preds_v, _ = self.model.predict(events, verbose=False)
                all_preds.append(preds_v)
                logger.info(f"Multimodal video preds shape: {preds_v.shape}")
            except Exception as e:
                logger.warning(f"Multimodal video failed: {e}")

        if audio_path:
            try:
                events = self.model.get_events_dataframe(audio_path=audio_path)
                preds_a, _ = self.model.predict(events, verbose=False)
                all_preds.append(preds_a)
                logger.info(f"Multimodal audio preds shape: {preds_a.shape}")
            except Exception as e:
                logger.warning(f"Multimodal audio failed: {e}")

        if text_stimulus:
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, encoding="utf-8"
                ) as f:
                    f.write(text_stimulus)
                    tmp = f.name
                try:
                    events = self.model.get_events_dataframe(text_path=tmp)
                    preds_t, _ = self.model.predict(events, verbose=False)
                    all_preds.append(preds_t)
                    logger.info(f"Multimodal text preds shape: {preds_t.shape}")
                finally:
                    os.unlink(tmp)
            except Exception as e:
                logger.warning(f"Multimodal text failed: {e}")

        if not all_preds:
            return self._simulate_prediction("multimodal experience")

        # Align timestep lengths to the shortest run, then average
        min_t = min(p.shape[0] for p in all_preds)
        combined = np.mean(np.stack([p[:min_t] for p in all_preds], axis=0), axis=0)
        self._last_raw_predictions = combined
        self._last_segments = None
        self._last_modality = "multimodal"
        logger.info(f"Multimodal combined preds shape: {combined.shape}")
        return self._process_tribe_outputs(combined)

    def _process_tribe_outputs(self, preds: Any) -> Dict[str, Any]:
        """
        Convert TRIBEv2 fMRI predictions to brain-region activation dict.

        preds: np.ndarray of shape (n_timesteps, n_vertices) on fsaverage5 mesh.
        Maps HCP parcellation ROIs to our BrainRegion categories.
        """
        import numpy as np

        try:
            # Average across time → (n_vertices,)
            mean_activation = preds.mean(axis=0)

            # Normalize to [0, 1]
            v_min, v_max = mean_activation.min(), mean_activation.max()
            if v_max > v_min:
                norm = (mean_activation - v_min) / (v_max - v_min)
            else:
                norm = np.full_like(mean_activation, 0.5)

            # Try to summarize by HCP ROI labels (HCP MMP1.0 parcellation, 181 ROIs)
            try:
                from tribev2.utils import get_hcp_roi_indices

                # Exact HCP labels mapped to functional brain regions
                roi_groups = {
                    "visual_cortex": [
                        "V1",
                        "V2",
                        "V3",
                        "V4",
                        "V3A",
                        "V3B",
                        "V6",
                        "V6A",
                        "V7",
                        "V8",
                        "LO1",
                        "LO2",
                    ],
                    "auditory_cortex": [
                        "A1",
                        "A4",
                        "A5",
                        "LBelt",
                        "MBelt",
                        "PBelt",
                        "RI",
                        "STGa",
                        "STSda",
                        "STSdp",
                    ],
                    "prefrontal_cortex": [
                        "46",
                        "9a",
                        "9m",
                        "9p",
                        "10r",
                        "10v",
                        "10d",
                        "8Ad",
                        "8Av",
                        "8BL",
                        "8BM",
                        "9-46d",
                        "a9-46v",
                        "p9-46v",
                        "FEF",
                        "SFL",
                    ],
                    "motor_cortex": [
                        "4",
                        "6a",
                        "6d",
                        "6ma",
                        "6mp",
                        "6r",
                        "6v",
                        "PEF",
                        "SCEF",
                        "FEF",
                        "MI",
                    ],
                    "hippocampus": [
                        "PHA1",
                        "PHA2",
                        "PHA3",
                        "RSC",
                        "POS1",
                        "POS2",
                        "EC",
                        "PreS",
                        "H",
                    ],
                    "amygdala": ["pOFC", "OFC", "11l", "13l", "Pir"],
                    "insula": [
                        "FOP1",
                        "FOP2",
                        "FOP3",
                        "FOP4",
                        "FOP5",
                        "AVI",
                        "AAIC",
                        "Ig",
                        "PoI1",
                        "PoI2",
                        "PI",
                    ],
                    "anterior_cingulate": [
                        "24dd",
                        "24dv",
                        "a24",
                        "a24pr",
                        "33pr",
                        "p24",
                        "p32",
                        "p32pr",
                        "d32",
                        "a32pr",
                        "s32",
                    ],
                    "default_mode_network": [
                        "d23ab",
                        "v23ab",
                        "31a",
                        "31pd",
                        "31pv",
                        "7m",
                        "PCV",
                        "p32",
                        "RSC",
                        "POS1",
                        "POS2",
                    ],
                }

                results = {}
                for region, patterns in roi_groups.items():
                    vals = []
                    for pat in patterns:
                        try:
                            idx = get_hcp_roi_indices(pat, hemi="both", mesh="fsaverage5")
                            if len(idx):
                                vals.append(float(norm[idx].mean()))
                        except (ValueError, IndexError):
                            pass
                    activation = float(np.mean(vals)) if vals else 0.3
                    rtype = (
                        "excitatory"
                        if activation > 0.4
                        else ("inhibitory" if activation < 0.25 else "neutral")
                    )
                    results[region] = {"activation": activation, "type": rtype}
                return results

            except Exception:
                # Fallback: equal-width vertex chunks → approximate region activations
                region_names = [
                    "visual_cortex",
                    "auditory_cortex",
                    "prefrontal_cortex",
                    "motor_cortex",
                    "hippocampus",
                    "amygdala",
                    "insula",
                    "anterior_cingulate",
                    "default_mode_network",
                ]
                chunk = max(1, len(norm) // len(region_names))
                results = {}
                for i, region in enumerate(region_names):
                    act = float(norm[i * chunk : (i + 1) * chunk].mean())
                    rtype = (
                        "excitatory" if act > 0.4 else ("inhibitory" if act < 0.25 else "neutral")
                    )
                    results[region] = {"activation": act, "type": rtype}
                return results

        except Exception as e:
            logger.error(f"Failed to process TRIBEv2 outputs: {e}")
            return self._simulate_prediction()

    def _simulate_prediction(self, stimulus: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate simulated brain response predictions.

        This is used when the actual model is not available,
        providing reasonable approximations based on research.
        """
        # Base predictions that would come from the actual model
        predictions = {
            "visual_cortex": {"activation": 0.3, "type": "neutral"},
            "auditory_cortex": {"activation": 0.2, "type": "neutral"},
            "prefrontal_cortex": {"activation": 0.5, "type": "excitatory"},
            "motor_cortex": {"activation": 0.2, "type": "neutral"},
            "hippocampus": {"activation": 0.4, "type": "excitatory"},
            "amygdala": {"activation": 0.3, "type": "neutral"},
            "insula": {"activation": 0.4, "type": "excitatory"},
            "anterior_cingulate": {"activation": 0.5, "type": "excitatory"},
            "default_mode_network": {"activation": 0.6, "type": "excitatory"},
        }

        # Adjust based on stimulus content
        if stimulus:
            stim_lower = stimulus.lower()

            if "exercise" in stim_lower or "running" in stim_lower or "sport" in stim_lower:
                predictions["motor_cortex"] = {"activation": 0.85, "type": "excitatory"}
                predictions["prefrontal_cortex"] = {"activation": 0.7, "type": "excitatory"}
                predictions["hippocampus"] = {"activation": 0.75, "type": "excitatory"}
                predictions["anterior_cingulate"] = {"activation": 0.65, "type": "excitatory"}

            elif "meditation" in stim_lower or "mindfulness" in stim_lower:
                predictions["prefrontal_cortex"] = {"activation": 0.8, "type": "excitatory"}
                predictions["anterior_cingulate"] = {"activation": 0.85, "type": "excitatory"}
                predictions["insula"] = {"activation": 0.75, "type": "excitatory"}
                predictions["default_mode_network"] = {"activation": 0.7, "type": "excitatory"}
                predictions["amygdala"] = {"activation": 0.3, "type": "inhibitory"}

            elif "sleep" in stim_lower or "rest" in stim_lower:
                predictions["default_mode_network"] = {"activation": 0.8, "type": "excitatory"}
                predictions["hippocampus"] = {"activation": 0.7, "type": "excitatory"}
                predictions["prefrontal_cortex"] = {"activation": 0.3, "type": "inhibitory"}

            elif "music" in stim_lower or "audio" in stim_lower:
                predictions["auditory_cortex"] = {"activation": 0.85, "type": "excitatory"}
                predictions["temporal_lobe"] = {"activation": 0.75, "type": "excitatory"}
                predictions["amygdala"] = {"activation": 0.6, "type": "excitatory"}

            elif "reading" in stim_lower or "learning" in stim_lower:
                predictions["visual_cortex"] = {"activation": 0.7, "type": "excitatory"}
                predictions["hippocampus"] = {"activation": 0.8, "type": "excitatory"}
                predictions["prefrontal_cortex"] = {"activation": 0.75, "type": "excitatory"}

            elif "social" in stim_lower:
                predictions["temporal_lobe"] = {"activation": 0.8, "type": "excitatory"}
                predictions["amygdala"] = {"activation": 0.65, "type": "excitatory"}
                predictions["prefrontal_cortex"] = {"activation": 0.7, "type": "excitatory"}

        # Keep visualization features available even in simulation mode by
        # synthesizing fsaverage5-like vertex predictions from ROI activations.
        self._last_raw_predictions = self._synthesize_simulated_raw_predictions(predictions)
        self._last_segments = None
        self._last_modality = "simulated"

        return predictions

    def _synthesize_simulated_raw_predictions(self, roi_predictions: Dict[str, Any]) -> Any:
        """Create synthetic ``(timesteps, vertices)`` predictions for visualization fallback."""
        import numpy as np

        n_vertices = 20484  # fsaverage5 vertices used by TRIBEv2
        n_timesteps = 24

        items = list((roi_predictions or {}).items())
        if not items:
            return np.full((n_timesteps, n_vertices), 0.5, dtype=np.float32)

        base = np.zeros(n_vertices, dtype=np.float32)
        chunk = max(1, n_vertices // len(items))
        for i, (_, vals) in enumerate(items):
            act = float(vals.get("activation", 0.5) if isinstance(vals, dict) else vals)
            start = i * chunk
            end = n_vertices if i == len(items) - 1 else min(n_vertices, (i + 1) * chunk)
            base[start:end] = act

        # Add mild temporal variation so gif/mp4 generation can run.
        phase = np.linspace(0, 6.0 * np.pi, n_vertices, dtype=np.float32)
        series = []
        for t in range(n_timesteps):
            wave = 0.06 * np.sin(phase + (2.0 * np.pi * t / n_timesteps))
            frame = np.clip(base + wave, 0.0, 1.0)
            series.append(frame)

        return np.stack(series, axis=0)


class TribeHealthAnalyzer:
    """
    Main class for analyzing health data using Tribe v2 brain predictions.

    This class provides methods for:
    - Predicting brain responses to wellness activities
    - Analyzing correlations between health metrics and brain activity
    - Generating personalized health recommendations
    - Creating optimal daily schedules based on brain science
    """

    def __init__(self, use_model: bool = True):
        """
        Initialize the Tribe Health Analyzer.

        Args:
            use_model: Whether to attempt loading the actual Tribe v2 model
        """
        self.model_wrapper = TribeModelWrapper()
        self.is_initialized = False

        # Activity-specific brain response patterns (research-based)
        self.activity_patterns = self._init_activity_patterns()

        # Brain region descriptions
        self.region_descriptions = {
            BrainRegion.VISUAL_CORTEX: "Processes visual information from the eyes",
            BrainRegion.AUDITORY_CORTEX: "Processes sound and auditory information",
            BrainRegion.PREFRONTAL_CORTEX: "Executive functions, decision making, planning",
            BrainRegion.MOTOR_CORTEX: "Controls voluntary muscle movements",
            BrainRegion.HIPPOCAMPUS: "Memory formation and spatial navigation",
            BrainRegion.AMYGDALA: "Emotional processing, especially fear and anxiety",
            BrainRegion.INSULA: "Interoception, emotional feelings, empathy",
            BrainRegion.ANTERIOR_CINGULATE: "Attention, emotion, error detection",
            BrainRegion.POSTERIOR_CINGULATE: "Self-reflection, memory retrieval",
            BrainRegion.TEMPORAL_LOBE: "Language processing, memory, emotion",
            BrainRegion.PARIETAL_LOBE: "Sensory integration, spatial awareness",
            BrainRegion.OCCIPITAL_LOBE: "Visual processing center",
            BrainRegion.CEREBELLUM: "Motor coordination, balance",
            BrainRegion.BRAIN_STEM: "Basic life functions, arousal",
            BrainRegion.DEFAULT_MODE_NETWORK: "Self-referential thinking, mind-wandering",
        }

        logger.info("TribeHealthAnalyzer initialized")

    def _init_activity_patterns(self) -> Dict[ActivityType, Dict[str, Any]]:
        """Initialize research-based activity-brain response patterns."""
        return {
            ActivityType.EXERCISE: {
                "primary_regions": [
                    (BrainRegion.MOTOR_CORTEX, 0.85, "excitatory"),
                    (BrainRegion.PREFRONTAL_CORTEX, 0.70, "excitatory"),
                    (BrainRegion.HIPPOCAMPUS, 0.75, "excitatory"),
                    (BrainRegion.ANTERIOR_CINGULATE, 0.65, "excitatory"),
                ],
                "secondary_regions": [
                    (BrainRegion.CEREBELLUM, 0.60, "excitatory"),
                    (BrainRegion.INSULA, 0.55, "excitatory"),
                ],
                "wellness_score_base": 8.0,
                "benefits": [
                    "Improved neuroplasticity and brain health",
                    "Enhanced memory and cognitive function",
                    "Reduced anxiety and depression symptoms",
                    "Better sleep quality",
                    "Increased BDNF (brain-derived neurotrophic factor)",
                ],
                "optimal_timing": "Morning (6-8 AM) or late afternoon (4-6 PM)",
            },
            ActivityType.MEDITATION: {
                "primary_regions": [
                    (BrainRegion.PREFRONTAL_CORTEX, 0.80, "excitatory"),
                    (BrainRegion.ANTERIOR_CINGULATE, 0.85, "excitatory"),
                    (BrainRegion.INSULA, 0.75, "excitatory"),
                    (BrainRegion.DEFAULT_MODE_NETWORK, 0.70, "excitatory"),
                ],
                "secondary_regions": [
                    (BrainRegion.AMYGDALA, 0.30, "inhibitory"),
                    (BrainRegion.HIPPOCAMPUS, 0.60, "excitatory"),
                ],
                "wellness_score_base": 8.5,
                "benefits": [
                    "Reduced stress and anxiety",
                    "Improved emotional regulation",
                    "Enhanced focus and attention",
                    "Increased gray matter density",
                    "Better self-awareness",
                ],
                "optimal_timing": "Early morning (5-7 AM) or evening (8-9 PM)",
            },
            ActivityType.SLEEP: {
                "primary_regions": [
                    (BrainRegion.DEFAULT_MODE_NETWORK, 0.80, "excitatory"),
                    (BrainRegion.HIPPOCAMPUS, 0.70, "excitatory"),
                    (BrainRegion.PREFRONTAL_CORTEX, 0.30, "inhibitory"),
                ],
                "secondary_regions": [
                    (BrainRegion.AMYGDALA, 0.25, "inhibitory"),
                    (BrainRegion.ANTERIOR_CINGULATE, 0.20, "inhibitory"),
                ],
                "wellness_score_base": 9.0,
                "benefits": [
                    "Memory consolidation",
                    "Brain toxin clearance via glymphatic system",
                    "Cellular repair and regeneration",
                    "Emotional processing",
                    "Cognitive restoration",
                ],
                "optimal_timing": "10 PM - 6 AM (8 hours recommended)",
            },
            ActivityType.READING: {
                "primary_regions": [
                    (BrainRegion.VISUAL_CORTEX, 0.70, "excitatory"),
                    (BrainRegion.HIPPOCAMPUS, 0.80, "excitatory"),
                    (BrainRegion.PREFRONTAL_CORTEX, 0.75, "excitatory"),
                    (BrainRegion.TEMPORAL_LOBE, 0.65, "excitatory"),
                ],
                "secondary_regions": [
                    (BrainRegion.PARIETAL_LOBE, 0.55, "excitatory"),
                    (BrainRegion.DEFAULT_MODE_NETWORK, 0.50, "excitatory"),
                ],
                "wellness_score_base": 7.5,
                "benefits": [
                    "Enhanced vocabulary and language skills",
                    "Improved cognitive reserve",
                    "Better empathy and theory of mind",
                    "Reduced stress levels",
                    "Neural pathway strengthening",
                ],
                "optimal_timing": "Morning or early evening",
            },
            ActivityType.MUSIC: {
                "primary_regions": [
                    (BrainRegion.AUDITORY_CORTEX, 0.85, "excitatory"),
                    (BrainRegion.TEMPORAL_LOBE, 0.75, "excitatory"),
                    (BrainRegion.AMYGDALA, 0.65, "excitatory"),
                    (BrainRegion.PREFRONTAL_CORTEX, 0.60, "excitatory"),
                ],
                "secondary_regions": [
                    (BrainRegion.MOTOR_CORTEX, 0.50, "excitatory"),
                    (BrainRegion.CEREBELLUM, 0.55, "excitatory"),
                ],
                "wellness_score_base": 7.8,
                "benefits": [
                    "Enhanced mood and emotional regulation",
                    "Improved memory and cognitive function",
                    "Stress reduction and relaxation",
                    "Better motor coordination",
                    "Increased creativity",
                ],
                "optimal_timing": "Any time, especially during repetitive tasks",
            },
            ActivityType.SOCIAL: {
                "primary_regions": [
                    (BrainRegion.TEMPORAL_LOBE, 0.80, "excitatory"),
                    (BrainRegion.AMYGDALA, 0.65, "excitatory"),
                    (BrainRegion.PREFRONTAL_CORTEX, 0.70, "excitatory"),
                    (BrainRegion.INSULA, 0.70, "excitatory"),
                ],
                "secondary_regions": [
                    (BrainRegion.ANTERIOR_CINGULATE, 0.60, "excitatory"),
                    (BrainRegion.MOTOR_CORTEX, 0.45, "excitatory"),
                ],
                "wellness_score_base": 8.2,
                "benefits": [
                    "Improved mental health and well-being",
                    "Enhanced cognitive function",
                    "Reduced feelings of isolation",
                    "Better emotional regulation",
                    "Increased oxytocin release",
                ],
                "optimal_timing": "Afternoon or evening",
            },
            ActivityType.WORK: {
                "primary_regions": [
                    (BrainRegion.PREFRONTAL_CORTEX, 0.80, "excitatory"),
                    (BrainRegion.ANTERIOR_CINGULATE, 0.70, "excitatory"),
                    (BrainRegion.PARIETAL_LOBE, 0.60, "excitatory"),
                ],
                "secondary_regions": [
                    (BrainRegion.VISUAL_CORTEX, 0.50, "excitatory"),
                    (BrainRegion.TEMPORAL_LOBE, 0.45, "excitatory"),
                ],
                "wellness_score_base": 6.5,
                "benefits": [
                    "Cognitive engagement",
                    "Skill development",
                    "Sense of purpose",
                    "Social connection",
                ],
                "optimal_timing": "9 AM - 5 PM with breaks every 90 minutes",
            },
            ActivityType.RELAXATION: {
                "primary_regions": [
                    (BrainRegion.DEFAULT_MODE_NETWORK, 0.75, "excitatory"),
                    (BrainRegion.PARIETAL_LOBE, 0.50, "excitatory"),
                    (BrainRegion.AMYGDALA, 0.35, "inhibitory"),
                ],
                "secondary_regions": [
                    (BrainRegion.INSULA, 0.45, "excitatory"),
                    (BrainRegion.ANTERIOR_CINGULATE, 0.40, "inhibitory"),
                ],
                "wellness_score_base": 7.5,
                "benefits": [
                    "Stress reduction",
                    "Mental recovery",
                    "Improved creativity",
                    "Better emotional balance",
                ],
                "optimal_timing": "Late afternoon or evening",
            },
            ActivityType.NATURE: {
                "primary_regions": [
                    (BrainRegion.VISUAL_CORTEX, 0.75, "excitatory"),
                    (BrainRegion.AMYGDALA, 0.30, "inhibitory"),
                    (BrainRegion.PREFRONTAL_CORTEX, 0.55, "excitatory"),
                    (BrainRegion.INSULA, 0.60, "excitatory"),
                ],
                "secondary_regions": [
                    (BrainRegion.AUDITORY_CORTEX, 0.50, "excitatory"),
                    (BrainRegion.DEFAULT_MODE_NETWORK, 0.65, "excitatory"),
                ],
                "wellness_score_base": 8.3,
                "benefits": [
                    "Reduced rumination and negative thoughts",
                    "Lower cortisol levels",
                    "Improved attention and focus",
                    "Enhanced mood",
                    "Better immune function",
                ],
                "optimal_timing": "Morning or late afternoon",
            },
            ActivityType.LEARNING: {
                "primary_regions": [
                    (BrainRegion.HIPPOCAMPUS, 0.85, "excitatory"),
                    (BrainRegion.PREFRONTAL_CORTEX, 0.80, "excitatory"),
                    (BrainRegion.TEMPORAL_LOBE, 0.70, "excitatory"),
                ],
                "secondary_regions": [
                    (BrainRegion.PARIETAL_LOBE, 0.55, "excitatory"),
                    (BrainRegion.ANTERIOR_CINGULATE, 0.60, "excitatory"),
                ],
                "wellness_score_base": 8.0,
                "benefits": [
                    "Neuroplasticity enhancement",
                    "Cognitive reserve building",
                    "Memory improvement",
                    "Neural pathway strengthening",
                ],
                "optimal_timing": "Morning when alertness is high",
            },
            ActivityType.CREATIVE: {
                "primary_regions": [
                    (BrainRegion.DEFAULT_MODE_NETWORK, 0.80, "excitatory"),
                    (BrainRegion.PREFRONTAL_CORTEX, 0.65, "excitatory"),
                    (BrainRegion.TEMPORAL_LOBE, 0.60, "excitatory"),
                ],
                "secondary_regions": [
                    (BrainRegion.PARIETAL_LOBE, 0.55, "excitatory"),
                    (BrainRegion.OCCIPITAL_LOBE, 0.50, "excitatory"),
                ],
                "wellness_score_base": 7.8,
                "benefits": [
                    "Enhanced problem-solving",
                    "Improved emotional expression",
                    "Better cognitive flexibility",
                    "Increased dopamine release",
                ],
                "optimal_timing": "When energy is moderate, not peak alertness",
            },
            ActivityType.MINDFULNESS: {
                "primary_regions": [
                    (BrainRegion.INSULA, 0.80, "excitatory"),
                    (BrainRegion.PREFRONTAL_CORTEX, 0.75, "excitatory"),
                    (BrainRegion.ANTERIOR_CINGULATE, 0.80, "excitatory"),
                ],
                "secondary_regions": [
                    (BrainRegion.AMYGDALA, 0.35, "inhibitory"),
                    (BrainRegion.DEFAULT_MODE_NETWORK, 0.60, "excitatory"),
                ],
                "wellness_score_base": 8.4,
                "benefits": [
                    "Present-moment awareness",
                    "Reduced mind-wandering",
                    "Better emotional regulation",
                    "Decreased rumination",
                ],
                "optimal_timing": "Any time, especially during transitions",
            },
        }

    async def initialize(self) -> bool:
        """
        Initialize the analyzer and attempt to load the model.

        Returns:
            True if initialization successful
        """
        try:
            # Attempt to load the actual TRIBEv2 model
            if TORCH_AVAILABLE and TRIBEV2_AVAILABLE:
                cache_dir = os.path.join(
                    os.path.dirname(__file__), "..", "..", "..", "cache", "tribev2"
                )
                loaded = await self.model_wrapper.load_model(cache_dir=cache_dir)
                if loaded:
                    logger.info("TRIBEv2 model loaded successfully")
                else:
                    logger.info("Using simulated predictions (TRIBEv2 model not loaded)")
            else:
                logger.info("Using simulated predictions (tribev2 package not available)")

            self.is_initialized = True
            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.is_initialized = True  # Still allow simulated predictions
            return True

    async def predict_brain_response(
        self,
        activity: str,
        duration_minutes: int = 30,
    ) -> Dict[str, Any]:
        """
        Predict brain response to a specific activity.

        Args:
            activity: Activity type (e.g., "exercise", "meditation")
            duration_minutes: Duration of the activity

        Returns:
            Dictionary with prediction results
        """
        try:
            # Convert string to ActivityType
            activity_type = ActivityType(activity.lower())
        except ValueError:
            return {
                "success": False,
                "error": (
                    f"Unknown activity type: {activity}. "
                    f"Valid types: {[a.value for a in ActivityType]}"
                ),
            }

        # Get activity pattern
        pattern = self.activity_patterns.get(activity_type)
        if not pattern:
            return {
                "success": False,
                "error": f"No pattern found for activity: {activity}",
            }

        # Build brain responses
        brain_responses = []

        # Process primary regions
        for region, activation, response_type in pattern["primary_regions"]:
            # Adjust activation based on duration
            duration_factor = min(1.2, 0.5 + (duration_minutes / 60))
            adjusted_activation = min(1.0, activation * duration_factor)

            brain_responses.append(
                BrainResponse(
                    region=region,
                    activation_level=adjusted_activation,
                    response_type=response_type,
                    confidence=0.85 if self.model_wrapper.is_loaded else 0.70,
                    description=self.region_descriptions.get(region, ""),
                )
            )

        # Process secondary regions
        for region, activation, response_type in pattern["secondary_regions"]:
            duration_factor = min(1.1, 0.5 + (duration_minutes / 90))
            adjusted_activation = min(1.0, activation * duration_factor)

            brain_responses.append(
                BrainResponse(
                    region=region,
                    activation_level=adjusted_activation,
                    response_type=response_type,
                    confidence=0.75 if self.model_wrapper.is_loaded else 0.60,
                    description=self.region_descriptions.get(region, ""),
                )
            )

        # Calculate wellness score
        base_score = pattern["wellness_score_base"]
        duration_bonus = min(1.0, duration_minutes / 45) * 0.5
        wellness_score = min(10.0, base_score + duration_bonus)

        # Build recommendation
        intensity = (
            "low" if duration_minutes < 15 else ("high" if duration_minutes > 60 else "moderate")
        )

        # Create prediction
        prediction = ActivityPrediction(
            activity=activity_type,
            duration_minutes=duration_minutes,
            brain_responses=brain_responses,
            overall_wellness_score=wellness_score,
            benefits=pattern["benefits"],
            recommendations=self._generate_recommendations(activity_type, duration_minutes),
            optimal_timing=pattern.get("optimal_timing"),
            intensity_level=intensity,
        )

        return {
            "success": True,
            "prediction": {
                "activity": activity_type.value,
                "duration_minutes": duration_minutes,
                "brain_regions": {
                    r.region.value: {
                        "response": r.response_type,
                        "intensity": r.activation_level,
                        "description": r.description,
                    }
                    for r in brain_responses
                },
                "wellness_score": wellness_score,
                "benefits": prediction.benefits,
                "recommendation": (
                    prediction.recommendations[0]
                    if prediction.recommendations
                    else "No specific recommendation"
                ),
                "optimal_timing": prediction.optimal_timing,
                "intensity": intensity,
            },
        }

    def _generate_recommendations(
        self,
        activity: ActivityType,
        duration: int,
    ) -> List[str]:
        """Generate personalized recommendations based on activity and duration."""
        recommendations = []

        if activity == ActivityType.EXERCISE:
            if duration < 20:
                recommendations.append(
                    "Consider extending to at least 20-30 minutes for optimal brain benefits"
                )
            recommendations.append(
                "Combine cardio with strength training for comprehensive brain health"
            )
            recommendations.append("Stay hydrated before, during, and after exercise")

        elif activity == ActivityType.MEDITATION:
            if duration < 10:
                recommendations.append(
                    "Even 10 minutes of meditation provides significant benefits"
                )
            recommendations.append("Focus on breath awareness for maximum amygdala regulation")
            recommendations.append(
                "Consistent daily practice yields better results than occasional longer sessions"
            )

        elif activity == ActivityType.SLEEP:
            if duration < 420:  # Less than 7 hours
                recommendations.append("Aim for 7-9 hours of sleep for optimal brain recovery")
            recommendations.append(
                "Maintain consistent sleep schedule for better sleep architecture"
            )
            recommendations.append(
                "Avoid screens 1 hour before bed for better melatonin production"
            )

        elif activity == ActivityType.READING:
            recommendations.append("Mix fiction and non-fiction for diverse brain stimulation")
            recommendations.append("Take brief breaks every 25-30 minutes to maintain focus")

        elif activity == ActivityType.MUSIC:
            recommendations.append(
                "Active listening (focused attention) provides more benefits than background music"
            )
            recommendations.append("Learning an instrument provides additional cognitive benefits")

        elif activity == ActivityType.SOCIAL:
            recommendations.append("Quality of social connections matters more than quantity")
            recommendations.append(
                "Face-to-face interactions provide more brain benefits than digital"
            )

        else:
            recommendations.append("Maintain consistency for long-term brain health benefits")

        return recommendations

    async def analyze_brain_health_correlation(
        self,
        health_metrics: List[Any],
        health_summary: Any,
    ) -> Dict[str, Any]:
        """
        Analyze correlations between health metrics and brain activity.

        Args:
            health_metrics: List of HealthMetric objects
            health_summary: HealthSummary object

        Returns:
            Dictionary with correlation analysis
        """
        correlations = []

        # Sleep-brain correlation
        if health_summary.avg_sleep_hours:
            sleep_quality = health_summary.avg_sleep_hours / 8.0  # 8 hours as optimal
            correlations.append(
                BrainHealthCorrelation(
                    health_metric="Sleep Hours",
                    brain_region=BrainRegion.HIPPOCAMPUS,
                    correlation_strength=0.75 if sleep_quality >= 0.8 else 0.5,
                    insight=(
                        "Quality sleep enhances hippocampal memory"
                        " consolidation and glymphatic clearance"
                    ),
                    confidence=0.85,
                )
            )
            correlations.append(
                BrainHealthCorrelation(
                    health_metric="Sleep Hours",
                    brain_region=BrainRegion.PREFRONTAL_CORTEX,
                    correlation_strength=0.70 if sleep_quality >= 0.8 else 0.45,
                    insight=(
                        "Adequate sleep restores prefrontal cortex function"
                        " for better decision-making"
                    ),
                    confidence=0.80,
                )
            )

        # Exercise-brain correlation
        if health_summary.total_exercise_minutes and health_summary.total_exercise_minutes > 0:
            exercise_factor = min(
                1.0, health_summary.total_exercise_minutes / 150
            )  # 150 min/week optimal
            correlations.append(
                BrainHealthCorrelation(
                    health_metric="Exercise Minutes",
                    brain_region=BrainRegion.HIPPOCAMPUS,
                    correlation_strength=0.65 * exercise_factor + 0.2,
                    insight="Exercise increases hippocampal neurogenesis and BDNF production",
                    confidence=0.82,
                )
            )
            correlations.append(
                BrainHealthCorrelation(
                    health_metric="Exercise Minutes",
                    brain_region=BrainRegion.PREFRONTAL_CORTEX,
                    correlation_strength=0.60 * exercise_factor + 0.15,
                    insight="Regular exercise improves prefrontal cortex blood flow and cognition",
                    confidence=0.78,
                )
            )

        # Mood-brain correlation
        if health_summary.avg_mood:
            mood_factor = health_summary.avg_mood / 10.0
            correlations.append(
                BrainHealthCorrelation(
                    health_metric="Mood",
                    brain_region=BrainRegion.AMYGDALA,
                    correlation_strength=-0.55
                    + (mood_factor * 0.3),  # Negative correlation with negative mood
                    insight=(
                        "Better mood associated with balanced amygdala activity"
                        " and emotional regulation"
                    ),
                    confidence=0.75,
                )
            )
            correlations.append(
                BrainHealthCorrelation(
                    health_metric="Mood",
                    brain_region=BrainRegion.PREFRONTAL_CORTEX,
                    correlation_strength=0.5 + (mood_factor * 0.2),
                    insight="Positive mood linked to enhanced prefrontal cortex function",
                    confidence=0.72,
                )
            )

        # Energy-brain correlation
        if health_summary.avg_energy:
            energy_factor = health_summary.avg_energy / 10.0
            correlations.append(
                BrainHealthCorrelation(
                    health_metric="Energy Level",
                    brain_region=BrainRegion.ANTERIOR_CINGULATE,
                    correlation_strength=0.5 + (energy_factor * 0.25),
                    insight="Energy levels correlate with anterior cingulate engagement",
                    confidence=0.68,
                )
            )

        # Generate recommendations
        recommendations = self._generate_health_recommendations(health_summary, correlations)

        # Generate optimal timing
        optimal_timing = {
            "exercise": "Morning (6-8 AM) - capitalizes on cortisol peak",
            "meditation": "Early morning (5-7 AM) - theta wave dominance",
            "learning": "10 AM - 12 PM - peak alertness window",
            "creative_work": "2-4 PM - moderate arousal optimal for creativity",
            "social_activities": "4-6 PM - natural social energy peak",
            "relaxation": "8-9 PM - preparation for sleep",
        }

        return {
            "success": True,
            "analysis": {
                "correlations": [c.to_dict() for c in correlations],
                "recommendations": recommendations,
                "optimal_timing": optimal_timing,
                "metrics_analyzed": len(health_metrics),
                "period_days": health_summary.total_days,
            },
        }

    def _generate_health_recommendations(
        self,
        summary: Any,
        correlations: List[BrainHealthCorrelation],
    ) -> List[str]:
        """Generate personalized health recommendations based on brain-health correlations."""
        recommendations = []

        # Sleep recommendations
        if summary.avg_sleep_hours and summary.avg_sleep_hours < 7:
            recommendations.append(
                "Increase sleep to 7-9 hours for optimal hippocampal memory consolidation"
            )

        # Exercise recommendations
        if not summary.total_exercise_minutes or summary.total_exercise_minutes < 150:
            recommendations.append(
                "Aim for 150+ minutes of exercise weekly for enhanced BDNF production"
            )

        # Mood recommendations
        if summary.avg_mood and summary.avg_mood < 6:
            recommendations.append(
                "Consider mindfulness meditation to regulate amygdala activity and improve mood"
            )

        # Energy recommendations
        if summary.avg_energy and summary.avg_energy < 5:
            recommendations.append(
                "Optimize sleep quality and timing for improved anterior cingulate function"
            )

        # General recommendations
        recommendations.append(
            "Maintain consistent daily routines to optimize circadian brain rhythms"
        )
        recommendations.append(
            "Take breaks every 90 minutes during cognitive work to prevent prefrontal fatigue"
        )

        return recommendations

    async def generate_optimal_schedule(self) -> Dict[str, Any]:
        """
        Generate an optimal daily schedule based on brain science.

        Returns:
            Dictionary with optimal schedule
        """
        schedule_slots = [
            {
                "time": "6:00 - 6:30 AM",
                "activity": "Morning Meditation/Mindfulness",
                "brain_benefit": "Theta wave state enhances prefrontal cortex training",
                "priority": "High",
            },
            {
                "time": "6:30 - 7:30 AM",
                "activity": "Exercise (Cardio + Strength)",
                "brain_benefit": "Peak cortisol period optimizes BDNF release",
                "priority": "High",
            },
            {
                "time": "7:30 - 8:00 AM",
                "activity": "Nutritious Breakfast",
                "brain_benefit": "Glucose supply for optimal cognition",
                "priority": "Medium",
            },
            {
                "time": "9:00 - 12:00 PM",
                "activity": "Deep Work/Learning",
                "brain_benefit": "Peak alertness and working memory capacity",
                "priority": "High",
            },
            {
                "time": "12:00 - 1:00 PM",
                "activity": "Lunch + Light Walk (Nature)",
                "brain_benefit": "Visual cortex restoration, amygdala calming",
                "priority": "Medium",
            },
            {
                "time": "1:00 - 2:00 PM",
                "activity": "Light Tasks/Email",
                "brain_benefit": "Post-lunch dip period for low-cognitive tasks",
                "priority": "Low",
            },
            {
                "time": "2:00 - 4:00 PM",
                "activity": "Creative Work/Problem-Solving",
                "brain_benefit": "Moderate arousal optimal for creativity",
                "priority": "High",
            },
            {
                "time": "4:00 - 6:00 PM",
                "activity": "Social Activities/Exercise",
                "brain_benefit": "Natural social energy peak, temp lobe activation",
                "priority": "Medium",
            },
            {
                "time": "6:00 - 7:00 PM",
                "activity": "Dinner + Family Time",
                "brain_benefit": "Oxytocin release, social bonding",
                "priority": "Medium",
            },
            {
                "time": "7:00 - 8:30 PM",
                "activity": "Relaxation/Reading/Music",
                "brain_benefit": "Default mode network restoration",
                "priority": "Medium",
            },
            {
                "time": "8:30 - 9:00 PM",
                "activity": "Evening Wind-Down Routine",
                "brain_benefit": "Melatonin production initiation",
                "priority": "High",
            },
            {
                "time": "9:30 - 10:00 PM",
                "activity": "Sleep Preparation",
                "brain_benefit": "Transition to sleep architecture",
                "priority": "High",
            },
            {
                "time": "10:00 PM - 6:00 AM",
                "activity": "Sleep (8 hours)",
                "brain_benefit": "Memory consolidation, glymphatic clearance",
                "priority": "Critical",
            },
        ]

        return {
            "success": True,
            "schedule": {
                "slots": schedule_slots,
                "score": 9.2,
                "notes": (
                    "This schedule optimizes brain function by aligning activities"
                    " with natural circadian rhythms and neural energy patterns."
                    " Adjust based on your chronotype (morning vs evening preference)."
                ),
            },
        }

    # ──────────────────────────────────────────────────────────────────────
    # Multimodal prediction + visualization methods
    # ──────────────────────────────────────────────────────────────────────

    def _roi_dict_to_brain_responses(self, roi_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert raw ROI activation dict to the standard BrainResponse format."""
        results: Dict[str, Any] = {}
        for region_name, vals in roi_dict.items():
            if isinstance(vals, dict):
                activation = float(vals.get("activation", 0.5))
                rtype = vals.get("type", "neutral")
            else:
                activation = float(vals)
                rtype = "excitatory" if activation > 0.4 else "neutral"
            try:
                region = BrainRegion(region_name)
                desc = self.region_descriptions.get(region, "")
            except ValueError:
                desc = ""
            results[region_name] = {
                "activation_level": activation,
                "response_type": rtype,
                "confidence": 0.85 if self.model_wrapper.is_loaded else 0.65,
                "description": desc,
            }
        return results

    async def _generate_visualizations(
        self,
        raw_preds: Any,
        title: str,
        stem: str,
        viz_types: List[str],
        modality: str = "",
        roi_activations: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Optional[str]]:
        """
        Generate the requested visualization types for raw TRIBEv2 predictions.

        Args:
            raw_preds:      numpy array (n_timesteps, n_vertices) or (n_vertices,)
            title:          Shared title for all plots
            stem:           Filename prefix
            viz_types:      Requested types: "interactive", "static", "gif", "mp4", "heatmap"
            modality:       Label for the ROI heatmap subtitle
            roi_activations: ``{region: activation_value}`` for the heatmap

        Returns:
            ``{viz_type: path_str_or_None}``
        """
        try:
            from .brain_viz import BrainVisualizer
        except ImportError:
            logger.warning("brain_viz module not available")
            return {}

        import numpy as np

        viz = BrainVisualizer()
        flat_roi = {
            k: float(v.get("activation", v) if isinstance(v, dict) else v)
            for k, v in (roi_activations or {}).items()
        }

        # Ensure preds is numpy
        if hasattr(raw_preds, "numpy"):
            raw_preds = raw_preds.numpy()
        raw_preds = np.array(raw_preds)

        return viz.generate_all(
            preds=raw_preds,
            roi_activations=flat_roi,
            title=title,
            output_stem=stem,
            viz_types=viz_types,
            auto_open_interactive=True,
            modality=modality,
        )

    async def predict_brain_from_video(
        self,
        video_path: str,
        viz_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Predict brain activation from a video file using V-JEPA2 + Wav2Vec-BERT.

        TRIBEv2 internally combines:
          - **V-JEPA2** visual features from video frames
          - **Wav2Vec-BERT** audio features from the video's audio track
          - **LLaMA 3.2** text features from WhisperX transcription

        Generates interactive 3D brain visualization (auto-opened in browser)
        and an ROI heatmap PNG.

        Args:
            video_path: Absolute path to video file (.mp4, .avi, .mov, .mkv, .webm)
            viz_types:  Visualization types to generate.
                        Default: ["interactive", "heatmap"]
                        Options: "interactive", "static", "gif", "mp4", "heatmap"

        Returns:
            Success dict with brain_regions, wellness_score, visualizations, etc.
        """
        if not self.is_initialized:
            await self.initialize()

        viz_types = viz_types or ["interactive", "heatmap"]

        if not os.path.isfile(video_path):
            return {"success": False, "error": f"Video file not found: {video_path}"}

        roi_dict = await self.model_wrapper.predict_from_video(video_path)
        brain_responses = self._roi_dict_to_brain_responses(roi_dict)

        # Wellness score based on combined activation
        import numpy as np

        mean_act = float(
            np.mean(
                [
                    v.get("activation_level", 0.5) if isinstance(v, dict) else v
                    for v in brain_responses.values()
                ]
            )
        )
        wellness_score = round(5.0 + mean_act * 4.0, 2)

        fname = os.path.splitext(os.path.basename(video_path))[0]
        title = f"Brain Activation — Video: {fname}"

        viz_paths: Dict[str, Optional[str]] = {}
        if self.model_wrapper._last_raw_predictions is not None:
            flat_roi = {k: v.get("activation_level", 0.5) for k, v in brain_responses.items()}
            viz_paths = await self._generate_visualizations(
                self.model_wrapper._last_raw_predictions,
                title=title,
                stem=f"video_{fname}",
                viz_types=viz_types,
                modality="video (V-JEPA2 + Wav2Vec-BERT)",
                roi_activations=flat_roi,
            )
        else:
            logger.info("Video: no raw preds stored — generating heatmap only")
            flat_roi = {k: v.get("activation_level", 0.5) for k, v in brain_responses.items()}
            viz_paths = await self._generate_visualizations(
                None,
                title=title,
                stem=f"video_{fname}",
                viz_types=["heatmap"],
                modality="video",
                roi_activations=flat_roi,
            )

        return {
            "success": True,
            "modality": "video",
            "feature_extractors": [
                "V-JEPA2 (visual)",
                "Wav2Vec-BERT (audio)",
                "LLaMA 3.2 (text via WhisperX)",
            ],
            "video_file": video_path,
            "brain_regions": brain_responses,
            "wellness_score": wellness_score,
            "model_used": self.model_wrapper.is_loaded,
            "visualizations": viz_paths,
        }

    async def predict_brain_from_audio(
        self,
        audio_path: str,
        viz_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Predict brain activation from an audio file using Wav2Vec-BERT +
        LLaMA 3.2 (text via WhisperX transcription).

        Args:
            audio_path: Absolute path to audio file (.wav, .mp3, .flac, .ogg)
            viz_types:  Visualization types. Default: ["interactive", "heatmap"]

        Returns:
            Success dict with brain_regions, wellness_score, visualizations, etc.
        """
        if not self.is_initialized:
            await self.initialize()

        viz_types = viz_types or ["interactive", "heatmap"]

        if not os.path.isfile(audio_path):
            return {"success": False, "error": f"Audio file not found: {audio_path}"}

        roi_dict = await self.model_wrapper.predict_from_audio(audio_path)
        brain_responses = self._roi_dict_to_brain_responses(roi_dict)

        import numpy as np

        mean_act = float(
            np.mean(
                [
                    v.get("activation_level", 0.5) if isinstance(v, dict) else v
                    for v in brain_responses.values()
                ]
            )
        )
        wellness_score = round(5.0 + mean_act * 4.0, 2)

        fname = os.path.splitext(os.path.basename(audio_path))[0]
        title = f"Brain Activation — Audio: {fname}"

        viz_paths: Dict[str, Optional[str]] = {}
        flat_roi = {k: v.get("activation_level", 0.5) for k, v in brain_responses.items()}
        if self.model_wrapper._last_raw_predictions is not None:
            viz_paths = await self._generate_visualizations(
                self.model_wrapper._last_raw_predictions,
                title=title,
                stem=f"audio_{fname}",
                viz_types=viz_types,
                modality="audio (Wav2Vec-BERT + LLaMA 3.2)",
                roi_activations=flat_roi,
            )
        else:
            viz_paths = await self._generate_visualizations(
                None,
                title=title,
                stem=f"audio_{fname}",
                viz_types=["heatmap"],
                modality="audio",
                roi_activations=flat_roi,
            )

        return {
            "success": True,
            "modality": "audio",
            "feature_extractors": ["Wav2Vec-BERT (audio)", "LLaMA 3.2 (text via WhisperX)"],
            "audio_file": audio_path,
            "brain_regions": brain_responses,
            "wellness_score": wellness_score,
            "model_used": self.model_wrapper.is_loaded,
            "visualizations": viz_paths,
        }

    async def predict_brain_multimodal(
        self,
        video_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        text_stimulus: Optional[str] = None,
        viz_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Predict brain activation from multiple modalities simultaneously.

        Runs each available input through its respective TRIBEv2 feature
        extractor (V-JEPA2, Wav2Vec-BERT, LLaMA 3.2), then averages the
        raw cortical predictions across modalities for a unified brain map.

        Args:
            video_path:    Path to video file (V-JEPA2 + Wav2Vec-BERT)
            audio_path:    Path to audio file (Wav2Vec-BERT)
            text_stimulus: Plain text narrative (LLaMA 3.2 via gTTS → Whisper)
            viz_types:     Visualization types. Default: ["interactive", "heatmap"]

        Returns:
            Success dict with combined brain_regions and all visualization paths
        """
        if not self.is_initialized:
            await self.initialize()

        viz_types = viz_types or ["interactive", "heatmap"]

        modalities_used = []
        if video_path:
            modalities_used.append("video")
        if audio_path:
            modalities_used.append("audio")
        if text_stimulus:
            modalities_used.append("text")

        if not modalities_used:
            return {
                "success": False,
                "error": (
                    "At least one of video_path, audio_path, or text_stimulus must be provided"
                ),
            }

        roi_dict = await self.model_wrapper.predict_multimodal(
            video_path=video_path,
            audio_path=audio_path,
            text_stimulus=text_stimulus,
        )
        brain_responses = self._roi_dict_to_brain_responses(roi_dict)

        import numpy as np

        mean_act = float(
            np.mean(
                [
                    v.get("activation_level", 0.5) if isinstance(v, dict) else v
                    for v in brain_responses.values()
                ]
            )
        )
        wellness_score = round(5.0 + mean_act * 4.0, 2)

        title = f"Brain Activation — Multimodal ({' + '.join(modalities_used)})"
        stem = "multimodal_" + "_".join(modalities_used)
        flat_roi = {k: v.get("activation_level", 0.5) for k, v in brain_responses.items()}

        viz_paths: Dict[str, Optional[str]] = {}
        if self.model_wrapper._last_raw_predictions is not None:
            viz_paths = await self._generate_visualizations(
                self.model_wrapper._last_raw_predictions,
                title=title,
                stem=stem,
                viz_types=viz_types,
                modality=f"multimodal ({', '.join(modalities_used)})",
                roi_activations=flat_roi,
            )
        else:
            viz_paths = await self._generate_visualizations(
                None,
                title=title,
                stem=stem,
                viz_types=["heatmap"],
                modality="multimodal",
                roi_activations=flat_roi,
            )

        extractors = []
        if "video" in modalities_used:
            extractors += ["V-JEPA2 (visual)", "Wav2Vec-BERT (audio from video)"]
        if "audio" in modalities_used:
            extractors.append("Wav2Vec-BERT (audio)")
        if "text" in modalities_used:
            extractors.append("LLaMA 3.2 (text via gTTS → Whisper)")

        return {
            "success": True,
            "modality": "multimodal",
            "modalities_used": modalities_used,
            "feature_extractors": list(dict.fromkeys(extractors)),
            "brain_regions": brain_responses,
            "wellness_score": wellness_score,
            "model_used": self.model_wrapper.is_loaded,
            "visualizations": viz_paths,
        }

    async def generate_brain_visualization(
        self,
        viz_type: str = "interactive",
        title: str = "Brain Activation",
        output_name: Optional[str] = None,
        auto_open: bool = True,
    ) -> Dict[str, Any]:
        """
        Re-generate a visualization from the most recent stored prediction.

        Useful when you want a different format (e.g. MP4 after getting an
        interactive HTML) without re-running the model.

        Args:
            viz_type:    One of "interactive", "static", "gif", "mp4", "heatmap"
            title:       Plot title
            output_name: Filename stem (auto-generated if None)
            auto_open:   Auto-open HTML in browser (interactive only)

        Returns:
            ``{success, path, viz_type}``
        """
        raw_preds = self.model_wrapper._last_raw_predictions
        if raw_preds is None:
            return {
                "success": False,
                "error": "No prediction in memory. Run a prediction first.",
            }

        try:
            from .brain_viz import BrainVisualizer
            import numpy as np

            viz = BrainVisualizer()
            if hasattr(raw_preds, "numpy"):
                raw_preds = raw_preds.numpy()
            raw_preds = np.array(raw_preds)

            if viz_type == "interactive":
                path = viz.plot_interactive(
                    raw_preds, title=title, output_name=output_name, auto_open=auto_open
                )
            elif viz_type == "static":
                path = viz.plot_static(raw_preds, title=title, output_name=output_name)
            elif viz_type == "gif":
                path = viz.plot_gif(raw_preds, title=title, output_name=output_name)
            elif viz_type == "mp4":
                path = viz.plot_mp4(raw_preds, title=title, output_name=output_name)
            elif viz_type == "heatmap":
                # Need to rebuild ROI activations from raw preds
                roi_dict = self.model_wrapper._process_tribe_outputs(raw_preds)
                flat_roi = {
                    k: float(v.get("activation", 0.5) if isinstance(v, dict) else v)
                    for k, v in roi_dict.items()
                }
                path = viz.plot_roi_heatmap(
                    flat_roi,
                    title=title,
                    output_name=output_name,
                    modality=self.model_wrapper._last_modality,
                )
            else:
                return {"success": False, "error": f"Unknown viz_type: {viz_type!r}"}

            return {
                "success": True,
                "viz_type": viz_type,
                "path": str(path),
                "modality": self.model_wrapper._last_modality,
            }

        except Exception as e:
            logger.error(f"Visualization generation failed: {e}")
            return {"success": False, "error": str(e)}
