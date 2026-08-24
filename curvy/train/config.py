"""Hyperparamètres d'entraînement.

Tout ce qui influence un résultat figure ici et est sérialisé dans le
checkpoint **et** dans la fiche d'expérience. Une expérience dont on ne peut
pas relire les hyperparamètres n'a pas eu lieu.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from curvy.config import DATASET_DIR, RUNS_DIR
from curvy.data.weighting import DEFAULT_DEPTH_TARGET
from curvy.seeding import DEFAULT_SEED


@dataclass
class TrainConfig:
    run_name: str = "exp-001"
    preset: str = "small"
    seed: int = DEFAULT_SEED

    steps: int = 20_000
    batch_size: int = 512
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    warmup_steps: int = 500
    min_lr_ratio: float = 0.05
    grad_clip: float = 1.0

    workers: int = 8
    bucketing: bool = False  # mesuré perdant aujourd'hui,
    bf16: bool = True
    compile_model: bool = False  # mesuré à x0,93 en Phase 0

    log_every: int = 50
    eval_every: int = 1_000
    ckpt_every: int = 1_000
    val_size: int = 512
    val_seed: int = 777  # figé et distinct de `seed` : le val ne bouge jamais

    skeletons: Path = DATASET_DIR / "skeletons-v1.jsonl.gz"
    #: Exclut du flux les squelettes réservés au jeu de test. Faux
    #: par défaut pour ne pas réécrire silencieusement l'histoire des runs
    #: exp-001 à exp-003, qui ont tourné sans. La valeur est écrite dans
    #: `config.json` : c'est elle qui dit si les chiffres d'un run sur le jeu
    #: de test veulent dire quelque chose.
    exclure_test: bool = False
    runs_dir: Path = RUNS_DIR
    depth_target: dict[int, float] = field(default_factory=lambda: dict(DEFAULT_DEPTH_TARGET))

    @property
    def run_dir(self) -> Path:
        return self.runs_dir / self.run_name

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
        return d
