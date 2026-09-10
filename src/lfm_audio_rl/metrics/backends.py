"""Optional scorers. Import dependencies only for an explicitly selected metric."""

import importlib
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from ..data import digest, file_hash
from .audio import resample


@contextmanager
def inference_seed(seed, device):
    # Localize stochastic cropping without changing the training process RNG.
    python_state, numpy_state = random.getstate(), np.random.get_state()
    target = torch.device(device)
    devices = (
        [target.index if target.index is not None else torch.cuda.current_device()]
        if target.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        try:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(devices[0]):
                    torch.cuda.manual_seed(seed)
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def asset_hash(path):
    path = Path(path).expanduser().resolve()
    if path.is_file():
        return file_hash(path)
    if not path.is_dir():
        raise ValueError(f"Scorer asset does not exist: {path}")
    files = sorted(
        p
        for p in path.rglob("*")
        if p.is_file() and not any(part.startswith(".") for part in p.relative_to(path).parts)
    )
    if not files:
        raise ValueError(f"Empty scorer directory: {path}")
    return digest({p.relative_to(path).as_posix(): file_hash(p) for p in files})


def scorer_assets(config):
    selected = {
        "speaker": ("speaker", config.speaker_model.path if config.speaker_model else None),
        "bertscore": ("bert", config.bert_model.path if config.bert_model else None),
        "fad": ("clap", config.clap_model.path if config.clap_model else None),
        "utmos": ("utmos", config.utmos_checkpoint),
        "nisqa": ("nisqa", config.nisqa_weights_dir),
        "dnsmos": ("dnsmos", config.dnsmos_weights_dir),
    }
    assets = {}
    for group in config.groups:
        if group in selected:
            name, path = selected[group]
            assets[name] = {
                "path": str(Path(path).expanduser().resolve()),
                "sha256": asset_hash(path),
            }
    if "reference" in config.groups and "visqol" in config.reference_metrics:
        assets["visqol"] = {
            "path": str(Path(config.visqol_model).expanduser().resolve()),
            "sha256": asset_hash(config.visqol_model),
        }
    if config.asr_model and Path(config.asr_model).is_dir():
        assets["asr"] = {
            "path": str(Path(config.asr_model).resolve()),
            "sha256": asset_hash(config.asr_model),
        }
    return assets


class Scorers:
    def __init__(self, config, cache_dir):
        self.config, self.cache_dir = config, Path(cache_dir)
        self.loaded = {}

    def _load(self, key, factory):
        if key not in self.loaded:
            self.loaded[key] = factory()
        return self.loaded[key]

    def speaker_embedding(self, wave, sr):
        from speechbrain.inference.speaker import EncoderClassifier

        source = str(Path(self.config.speaker_model.path).expanduser().resolve())
        model = self._load(
            "speaker",
            lambda: EncoderClassifier.from_hparams(
                source=source,
                savedir=str(self.cache_dir / "speaker"),
                overrides={"pretrained_path": source},
                run_opts={"device": self.config.device},
            ),
        )
        x = (
            torch.from_numpy(resample(wave, sr, 16000).astype(np.float32))
            .unsqueeze(0)
            .to(self.config.device)
        )
        with torch.inference_mode():
            embedding = model.encode_batch(x).detach().float().cpu().numpy().reshape(-1)
        return embedding

    def speaker_similarity(self, reference, reference_sr, wave, sr):
        a, b = self.speaker_embedding(reference, reference_sr), self.speaker_embedding(wave, sr)
        denominator = np.linalg.norm(a) * np.linalg.norm(b)
        if denominator <= 0:
            raise ValueError("Zero speaker embedding")
        return {"speaker_cosine": float(np.clip(a @ b / denominator, -1, 1))}

    def bertscore(self, reference, hypothesis):
        from bert_score import BERTScorer

        spec = self.config.bert_model
        model = self._load(
            "bert",
            lambda: BERTScorer(
                model_type=str(Path(spec.path).expanduser().resolve()),
                num_layers=spec.layers,
                idf=False,
                rescale_with_baseline=False,
                device=self.config.device,
            ),
        )
        precision, recall, f1 = model.score([hypothesis], [reference])
        return {
            "bertscore_precision": float(precision[0]),
            "bertscore_recall": float(recall[0]),
            "bertscore_f1": float(f1[0]),
        }

    def dnsmos(self, wave, sr):
        module = importlib.import_module("torchmetrics.functional.audio.dnsmos")
        path = Path(self.config.dnsmos_weights_dir).expanduser().resolve()
        for name in ["DNSMOS/model_v8.onnx", "DNSMOS/sig_bak_ovr.onnx"]:
            if not (path / name).is_file():
                raise ValueError(f"Missing local DNSMOS weight: {name}")
        module.DNSMOS_DIR = str(path)
        values = module.deep_noise_suppression_mean_opinion_score(
            torch.tensor(wave, dtype=torch.float32),
            sr,
            personalized=False,
            device="cpu",
            num_threads=1,
            cache_session=True,
        )
        return dict(
            zip(
                ["dnsmos_p808", "dnsmos_signal", "dnsmos_background", "dnsmos_overall"],
                values.detach().cpu().reshape(-1).tolist(),
                strict=True,
            )
        )

    def nisqa(self, wave, sr):
        module = importlib.import_module("torchmetrics.functional.audio.nisqa")
        path = Path(self.config.nisqa_weights_dir).expanduser().resolve()
        if not (path / "nisqa.tar").is_file():
            raise ValueError("Local NISQA directory must contain nisqa.tar")
        if self.loaded.get("nisqa_path") != str(path):
            module.NISQA_DIR = str(path)
            module._load_nisqa_model.cache_clear()
            self.loaded["nisqa_path"] = str(path)
        values = module.non_intrusive_speech_quality_assessment(
            torch.tensor(wave, dtype=torch.float32), sr
        )
        # The API order differs from some prose descriptions of the model.
        return dict(
            zip(
                [
                    "nisqa_overall",
                    "nisqa_noisiness",
                    "nisqa_discontinuity",
                    "nisqa_coloration",
                    "nisqa_loudness",
                ],
                values.detach().cpu().reshape(-1).tolist(),
                strict=True,
            )
        )

    def utmos(self, wave, sr):
        import utmosv2

        model = self._load(
            "utmos",
            lambda: utmosv2.create_model(
                pretrained=True,
                config="fusion_stage3",
                fold=0,
                seed=42,
                checkpoint_path=str(Path(self.config.utmos_checkpoint).expanduser().resolve()),
                device=self.config.device,
            ),
        )
        with inference_seed(self.config.seed, self.config.device):
            values = model.predict(
                data=wave.astype(np.float32),
                sr=sr,
                device=self.config.device,
                num_workers=0,
                num_repetitions=1,
                remove_silent_section=False,
                verbose=False,
            )
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        return {"utmos_v2": float(np.asarray(values).reshape(-1)[0])}

    def clap_embedding(self, wave, sr):
        from transformers import ClapModel, ClapProcessor

        path = str(Path(self.config.clap_model.path).expanduser().resolve())
        model = self._load(
            "clap",
            lambda: (
                ClapModel.from_pretrained(path, local_files_only=True).to(self.config.device).eval()
            ),
        )
        processor = self._load(
            "clap_processor", lambda: ClapProcessor.from_pretrained(path, local_files_only=True)
        )
        # Fixed 10-second chunks, including a shorter last chunk; no random crop
        # of a long utterance. The utterance embedding is the mean of its chunks.
        x = resample(wave, sr, 48000).astype(np.float32)
        chunks = [x[i : i + 480000] for i in range(0, len(x), 480000)]
        embeddings = []
        for chunk in chunks:
            with inference_seed(self.config.seed, self.config.device):
                batch = processor(audio=chunk, sampling_rate=48000, return_tensors="pt")
            batch = {k: v.to(self.config.device) for k, v in batch.items()}
            with torch.inference_mode():
                feature = model.get_audio_features(**batch)
            embeddings.append(feature.detach().float().cpu().numpy().reshape(-1))
        return np.mean(embeddings, axis=0)

    def intrusive(self, reference, reference_sr, wave, sr, relation):
        from .audio import aligned_scores, mel_cepstral_distortion

        a, b = resample(reference, reference_sr, 16000), resample(wave, sr, 16000)
        result, missing = {}, {}
        for name in self.config.reference_metrics:
            key = {
                "si_sdr": "si_sdr_db",
                "snr": "snr_db",
                "pesq": "pesq_wb",
                "visqol": "visqol_moslqo",
                "mcd": "mcd_dtw_db",
            }.get(name, name)
            if relation == "none" or (name != "mcd" and relation != "aligned"):
                missing[key] = (
                    "Requires declared aligned reference"
                    if name != "mcd"
                    else "Requires declared same-content reference"
                )
                continue
            if name != "mcd" and a.shape != b.shape:
                missing[key] = (
                    "Aligned signals have different lengths; no cropping or time warping applied"
                )
                continue
            if name in {"si_sdr", "snr"}:
                result[key] = aligned_scores(a, b)[key]
            elif name in {"stoi", "estoi"}:
                from pystoi import stoi

                if min(len(a), len(b)) < 16000:
                    missing[key] = "STOI/ESTOI require at least one second in this protocol"
                else:
                    result[key] = float(stoi(a, b, 16000, extended=name == "estoi"))
            elif name == "pesq":
                from pesq import pesq

                result[key] = float(pesq(16000, a, b, "wb"))
            elif name == "visqol":
                from visqol import visqol_lib_py
                from visqol.pb2 import visqol_config_pb2

                def load():
                    cfg = visqol_config_pb2.VisqolConfig()
                    cfg.audio.sample_rate = 16000
                    cfg.options.use_speech_scoring = True
                    cfg.options.svr_model_path = str(
                        Path(self.config.visqol_model).expanduser().resolve()
                    )
                    api = visqol_lib_py.VisqolApi()
                    api.Create(cfg)
                    return api

                api = self._load("visqol", load)
                result[key] = float(api.Measure(a.astype(np.float64), b.astype(np.float64)).moslqo)
            elif name == "mcd":
                result[key] = mel_cepstral_distortion(a, b, 16000)
        return result, missing
