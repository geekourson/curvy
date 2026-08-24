"""Flux d'entraînement : squelettes stratifiés + nuages tirés en ligne.

Rien n'est stocké sur disque à part la liste des squelettes. Chaque exemple est
fabriqué à la volée, si bien qu'un même squelette ne produit jamais deux fois
le même nuage — il n'y a donc pas de notion d'epoch, et pas de sur-apprentissage
au sens habituel.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from curvy.data.expr import from_prefix
from curvy.data.generate import load_skeletons
from curvy.data.pointcloud import CloudConfig, sample_cloud
from curvy.data.weighting import DEFAULT_DEPTH_TARGET, stratified_weights
from curvy.tokenizer.vocab import MAX_SEQ_LEN, PAD_ID, encode

__all__ = [
    "Batch",
    "BucketedBatches",
    "CurvyStream",
    "ValidationExample",
    "collate",
    "make_validation_set",
]


@dataclass
class Batch:
    points: torch.Tensor  # (B, N, 2) float32
    point_mask: torch.Tensor  # (B, N) bool — True = position de remplissage
    tokens: torch.Tensor  # (B, L) int64, <bos> … <eos> puis <pad>
    token_mask: torch.Tensor  # (B, L) bool — True = <pad>

    def to(self, device: torch.device) -> Batch:
        return Batch(
            self.points.to(device, non_blocking=True),
            self.point_mask.to(device, non_blocking=True),
            self.tokens.to(device, non_blocking=True),
            self.token_mask.to(device, non_blocking=True),
        )


class CurvyStream(IterableDataset):
    """Flux infini d'exemples (nuage, squelette encodé)."""

    def __init__(
        self,
        skeleton_path: Path,
        seed: int,
        cloud_cfg: CloudConfig | None = None,
        depth_target: dict[int, float] | None = None,
        max_seq_len: int = MAX_SEQ_LEN,
        max_retries: int = 12,
        exclure: frozenset[str] | None = None,
        garder: frozenset[str] | None = None,
    ) -> None:
        items = load_skeletons(skeleton_path)
        # `exclure` porte les squelettes du jeu de test (Phase 6) :
        # sans ce filtre, le jeu de test mesure de la restitution. `garder` fait
        # l'inverse et sert à construire le jeu de test lui-même.
        if exclure and garder:
            raise ValueError("`exclure` et `garder` sont exclusifs l'un de l'autre")
        self.n_exclus = 0
        if exclure:
            avant = len(items)
            items = [it for it in items if it["prefix"] not in exclure]
            self.n_exclus = avant - len(items)
        elif garder:
            avant = len(items)
            items = [it for it in items if it["prefix"] in garder]
            self.n_exclus = avant - len(items)
            if not items:
                raise ValueError("`garder` ne retient aucun squelette")
        self.prefixes = [it["prefix"] for it in items]
        self.depths = [it["depth"] for it in items]
        self.weights = stratified_weights(self.depths, depth_target or DEFAULT_DEPTH_TARGET)
        self.seed = seed
        self.cloud_cfg = cloud_cfg or CloudConfig()
        self.max_seq_len = max_seq_len
        self.max_retries = max_retries
        self._trees: list | None = None

    def _tree(self, i: int):
        if self._trees is None:
            self._trees = [None] * len(self.prefixes)
        if self._trees[i] is None:
            self._trees[i] = from_prefix(self.prefixes[i].split())
        return self._trees[i]

    def __iter__(self) -> Iterator[tuple[np.ndarray, list[int]]]:
        info = get_worker_info()
        wid = 0 if info is None else info.id
        rng = np.random.default_rng(self.seed + 7919 * wid)
        idx_pool = np.arange(len(self.prefixes))
        while True:
            i = int(rng.choice(idx_pool, p=self.weights))
            node = self._tree(i)
            for _ in range(self.max_retries):
                cloud, _ = sample_cloud(rng, node, self.cloud_cfg)
                if cloud is not None:
                    break
            else:
                continue  # squelette récalcitrant : on passe au suivant
            ids = encode(node)
            if len(ids) > self.max_seq_len:
                continue
            pts = np.stack([cloud.x, cloud.y], axis=1).astype(np.float32)
            yield pts, ids


def collate(samples: list[tuple[np.ndarray, list[int]]]) -> Batch:
    """Remplissage à la volée. Les masques valent True sur le remplissage."""
    n_max = max(len(p) for p, _ in samples)
    l_max = max(len(t) for _, t in samples)
    b = len(samples)

    points = torch.zeros(b, n_max, 2, dtype=torch.float32)
    point_mask = torch.ones(b, n_max, dtype=torch.bool)
    tokens = torch.full((b, l_max), PAD_ID, dtype=torch.long)
    token_mask = torch.ones(b, l_max, dtype=torch.bool)

    for i, (pts, ids) in enumerate(samples):
        points[i, : len(pts)] = torch.from_numpy(pts)
        point_mask[i, : len(pts)] = False
        tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        token_mask[i, : len(ids)] = False
    return Batch(points, point_mask, tokens, token_mask)


@dataclass
class ValidationExample:
    """Un exemple de validation, avec la **vérité terrain** nécessaire au R².

    ``y`` est ce que le modèle voit (bruité, normalisé) ; ``y_clean`` est la
    valeur exacte de la fonction aux mêmes abscisses, également normalisée.
    C'est contre ``y_clean`` que le R² se mesure (précision du
    2026-08-19) — mesurer contre ``y`` plafonnerait au niveau de bruit qu'on a
    soi-même injecté.
    """

    points: np.ndarray
    ids: list[int]
    node: object
    x: np.ndarray
    y: np.ndarray
    y_clean: np.ndarray
    depth: int


def make_validation_set(
    skeleton_path: Path,
    n: int,
    seed: int,
    cloud_cfg: CloudConfig | None = None,
    garder: frozenset[str] | None = None,
    un_nuage_par_squelette: bool = False,
) -> list[ValidationExample]:
    """Jeu de validation **figé** : mêmes exemples à chaque run, donc courbes
    comparables entre expériences.

    ``garder`` restreint le tirage à un sous-ensemble de squelettes — c'est
    ainsi qu'on bâtit le jeu de test de la Phase 6, à partir des seuls
    squelettes tenus à l'écart de l'entraînement.

    ``un_nuage_par_squelette`` parcourt les squelettes au lieu de les tirer :
    chaque squelette apparaît exactement une fois. Le tirage stratifié
    surreprésenterait sinon les strates profondes au sein d'une réserve déjà
    construite pour être équilibrée.

    Dans ce mode, chaque squelette a droit à ``max_retries`` tentatives de
    nuage, comme dans le flux d'entraînement. Une seule tentative perdrait
    **30 % des squelettes** (mesuré : 349/500 contre 492/500 à douze essais) —
    et pas les plus difficiles, seulement les moins chanceux du premier coup.
    Le jeu de test serait alors biaisé vers les squelettes commodes.
    """
    from curvy.data.expr import depth as tree_depth
    from curvy.data.expr import evaluate

    stream = CurvyStream(skeleton_path, seed=seed, cloud_cfg=cloud_cfg, garder=garder)
    rng = np.random.default_rng(seed)
    items = stream.prefixes
    weights = stream.weights
    out: list[ValidationExample] = []
    ordre = rng.permutation(len(items)) if un_nuage_par_squelette else None
    curseur = 0
    while len(out) < n:
        if ordre is not None:
            if curseur >= len(ordre):
                break  # réserve épuisée : on rend moins que `n`, et ça se voit
            i = int(ordre[curseur])
            curseur += 1
        else:
            i = int(rng.choice(len(items), p=weights))
        node = stream._tree(i)
        essais = stream.max_retries if un_nuage_par_squelette else 1
        cloud = None
        for _ in range(essais):
            cloud, _ = sample_cloud(rng, node, stream.cloud_cfg)
            if cloud is not None:
                break
        if cloud is None:
            continue
        ids = encode(node)
        if len(ids) > stream.max_seq_len:
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            clean = (evaluate(node, cloud.x, cloud.consts) - cloud.y_offset) / cloud.y_scale
        if not np.isfinite(clean).all():
            continue
        pts = np.stack([cloud.x, cloud.y], axis=1).astype(np.float32)
        out.append(ValidationExample(pts, ids, node, cloud.x, cloud.y, clean, tree_depth(node)))
    return out


class BucketedBatches(IterableDataset):
    """Regroupe les exemples de tailles voisines avant de former les batches.

    Le nombre de points est tiré uniformément dans [20, 200] : en remplissant
    chaque batch jusqu'à son maximum, **45,1 % des emplacements alloués sont du
    remplissage** (mesuré le 2026-08-19). Comme l'attention de l'encodeur coûte
    O(B·N²), on paie presque le double du nécessaire.

    **Tampon glissant, et non par blocs.** La première version tamponnait
    ``16 × batch_size`` exemples, les triait, puis livrait tous les batches d'un
    coup. Résultat mesuré : 73,7 % du temps passé à attendre les données, et un
    entraînement 2,3 fois plus lent qu'*avec* le remplissage inutile. Le
    générateur produit 404 exemples/s par cœur ; un tampon de 8192 met 20 s à se
    remplir, pendant lesquelles le GPU ne fait rien.

    Ici le tampon est rempli une seule fois, puis maintenu : chaque batch en
    retire ``batch_size`` exemples de longueurs voisines et en réinjecte autant.
    La consommation devient régulière, la rafale disparaît.

    Le décalage de départ est tiré au hasard à chaque batch : sans lui, le
    modèle verrait toujours les nuages les plus courts en premier, ce qui
    corrélerait la taille effective du batch au pas d'entraînement.
    """

    def __init__(self, stream: CurvyStream, batch_size: int, pool_factor: int = 8) -> None:
        self.stream = stream
        self.batch_size = batch_size
        self.pool_factor = pool_factor

    def __iter__(self) -> Iterator[Batch]:
        info = get_worker_info()
        wid = 0 if info is None else info.id
        rng = np.random.default_rng(self.stream.seed + 104729 * wid)
        src = iter(self.stream)
        pool_size = self.batch_size * self.pool_factor
        pool = [next(src) for _ in range(pool_size)]
        while True:
            pool.sort(key=lambda s: len(s[0]))
            start = int(rng.integers(0, len(pool) - self.batch_size + 1))
            batch = pool[start : start + self.batch_size]
            del pool[start : start + self.batch_size]
            pool.extend(next(src) for _ in range(self.batch_size))
            yield collate(batch)
