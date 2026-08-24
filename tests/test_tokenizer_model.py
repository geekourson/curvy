"""Tests du tokenizer (Phase 2) et de l'architecture (Phase 3).

Deux propriétés valent tous les autres tests de ce fichier :

- ``legal_mask`` ne doit **jamais** permettre une séquence invalide, quel que
  soit le chemin suivi. C'est ce qui garantit 0 % de sorties malformées en
  beam search ;
- le modèle doit être **invariant par permutation** des points d'entrée
 . Un nuage n'a pas d'ordre.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from curvy.data.expr import from_prefix, prefix_is_complete, to_prefix
from curvy.data.grammar import ID_TO_TOKEN, MAX_CONSTANTS
from curvy.data.sample import sample_skeleton
from curvy.model.config import PRESETS, ModelConfig
from curvy.model.curvy import CurvyModel, count_parameters
from curvy.seeding import make_rng, seed_everything
from curvy.tokenizer.vocab import (
    BOS_ID,
    CONST_ID,
    EOS_ID,
    MAX_SEQ_LEN,
    PAD_ID,
    VOCAB_SIZE,
    DecodeState,
    decode,
    encode,
    legal_mask,
)


def _skeletons(n=120, seed=3):
    rng = make_rng(seed)
    out = []
    while len(out) < n:
        sk = sample_skeleton(rng)
        if sk is not None:
            out.append(sk)
    return out


# --- tokenizer ----------------------------------------------------------------


def test_roundtrip_encode_decode():
    for sk in _skeletons():
        assert decode(encode(sk)) == sk


def test_encode_encadre_par_bos_eos():
    ids = encode(_skeletons(1)[0])
    assert ids[0] == BOS_ID and ids[-1] == EOS_ID


def test_decode_ignore_pad_et_bos_et_stoppe_a_eos():
    sk = from_prefix(["add", "mul", "C", "x", "C"])
    ids = [BOS_ID, *encode(sk, add_special=False), EOS_ID, PAD_ID, PAD_ID]
    assert decode(ids) == sk


def test_toutes_les_sequences_du_dataset_tiennent_dans_max_seq_len():
    for sk in _skeletons(300, seed=21):
        assert len(encode(sk)) <= MAX_SEQ_LEN


# --- masque d'arité (la propriété centrale) -----------------------------------


def test_masque_ne_permet_que_des_sequences_valides():
    """Marche aléatoire guidée uniquement par le masque : tout chemin doit
    aboutir à un arbre syntaxiquement complet."""
    rng = np.random.default_rng(0)
    for _ in range(400):
        state = DecodeState()
        ids: list[int] = []
        while True:
            mask = legal_mask(state)
            legal = np.flatnonzero(mask)
            assert len(legal) > 0, "impasse : le masque ne propose plus rien"
            tid = int(rng.choice(legal))
            if tid == EOS_ID:
                break
            ids.append(tid)
            state.advance(tid)
        toks = [ID_TO_TOKEN[i] for i in ids]
        assert prefix_is_complete(toks)
        from_prefix(toks)  # doit se parser sans exception


def test_eos_interdit_tant_que_l_arbre_est_incomplet():
    state = DecodeState()  # remaining == 1, rien d'émis
    assert not legal_mask(state)[EOS_ID]
    state.advance(CONST_ID)  # arbre complet : la feuille C suffit
    mask = legal_mask(state)
    assert mask[EOS_ID] and mask.sum() == 1, "une fois complet, seul <eos> est permis"


def test_budget_de_constantes_ferme_le_token_C():
    state = DecodeState()
    state.n_consts = MAX_CONSTANTS
    state.remaining = 2
    assert not legal_mask(state)[CONST_ID]


def test_budget_de_longueur_force_les_feuilles():
    """Près de la limite, seuls les tokens qui referment l'arbre restent."""
    state = DecodeState()
    state.remaining = 2
    state.n_emitted = MAX_SEQ_LEN - 4
    mask = legal_mask(state)
    for tid in np.flatnonzero(mask):
        assert ID_TO_TOKEN[int(tid)] in ("x", "C"), ID_TO_TOKEN[int(tid)]


# --- architecture --------------------------------------------------------------


def test_comptes_de_parametres_conformes_aux_cibles():
    counts = {name: count_parameters(CurvyModel(cfg))["TOTAL"] for name, cfg in PRESETS.items()}
    assert 4.5e6 < counts["small"] < 6e6, counts
    assert 28e6 < counts["v1"] < 34e6, counts


def test_forward_produit_la_bonne_forme():
    seed_everything(0)
    cfg = ModelConfig(
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.0,
    )
    model = CurvyModel(cfg).eval()
    b, n, ell = 3, 17, 9
    pts = torch.randn(b, n, 2)
    pmask = torch.zeros(b, n, dtype=torch.bool)
    toks = torch.randint(3, VOCAB_SIZE, (b, ell))
    tmask = torch.zeros(b, ell, dtype=torch.bool)
    out = model(pts, pmask, toks, tmask)
    assert out.shape == (b, ell, VOCAB_SIZE)
    assert torch.isfinite(out).all()


def test_invariance_par_permutation_des_points():
    """un nuage est un ensemble. Le permuter ne doit rien changer."""
    seed_everything(1)
    cfg = ModelConfig(
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        point_positional=False,
    )
    model = CurvyModel(cfg).eval()
    pts = torch.randn(2, 24, 2)
    pmask = torch.zeros(2, 24, dtype=torch.bool)
    toks = torch.randint(3, VOCAB_SIZE, (2, 7))
    tmask = torch.zeros(2, 7, dtype=torch.bool)
    with torch.no_grad():
        a = model(pts, pmask, toks, tmask)
        perm = torch.randperm(24)
        b = model(pts[:, perm], pmask[:, perm], toks, tmask)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max().item()


def test_encodage_positionnel_brise_l_invariance():
    """Contre-épreuve : avec encodage positionnel, l'invariance doit disparaître.

    Sans ce test, un bug qui désactiverait silencieusement l'encodage
    positionnel passerait inaperçu et l'ablation de la Phase 4 comparerait deux
    fois la même chose."""
    seed_everything(1)
    cfg = ModelConfig(
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        point_positional=True,
    )
    model = CurvyModel(cfg).eval()
    pts = torch.randn(1, 24, 2)
    pmask = torch.zeros(1, 24, dtype=torch.bool)
    toks = torch.randint(3, VOCAB_SIZE, (1, 7))
    tmask = torch.zeros(1, 7, dtype=torch.bool)
    with torch.no_grad():
        a = model(pts, pmask, toks, tmask)
        b = model(pts[:, torch.randperm(24)], pmask, toks, tmask)
    assert not torch.allclose(a, b, atol=1e-4)


def test_le_masque_causal_empeche_de_voir_le_futur():
    seed_everything(2)
    cfg = ModelConfig(
        d_model=64,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.0,
    )
    model = CurvyModel(cfg).eval()
    pts = torch.randn(1, 10, 2)
    pmask = torch.zeros(1, 10, dtype=torch.bool)
    toks = torch.randint(3, VOCAB_SIZE, (1, 8))
    tmask = torch.zeros(1, 8, dtype=torch.bool)
    with torch.no_grad():
        ref = model(pts, pmask, toks, tmask)
        altered = toks.clone()
        altered[0, -1] = (altered[0, -1] + 1) % VOCAB_SIZE
        got = model(pts, pmask, altered, tmask)
    # Modifier le dernier token ne doit rien changer avant lui.
    assert torch.allclose(ref[:, :-1], got[:, :-1], atol=1e-5)


def test_fourier_et_lineaire_produisent_tous_deux_un_forward_valide():
    for mode in ("linear", "fourier"):
        cfg = ModelConfig(
            d_model=32,
            n_heads=2,
            n_encoder_layers=1,
            n_decoder_layers=1,
            dim_feedforward=64,
            dropout=0.0,
            point_encoding=mode,
        )
        model = CurvyModel(cfg).eval()
        out = model(
            torch.randn(2, 11, 2),
            torch.zeros(2, 11, dtype=torch.bool),
            torch.randint(3, VOCAB_SIZE, (2, 5)),
            torch.zeros(2, 5, dtype=torch.bool),
        )
        assert torch.isfinite(out).all()
    with pytest.raises(ValueError):
        CurvyModel(ModelConfig(point_encoding="magique"))


def test_encode_decode_survit_a_un_aller_retour_par_les_ids():
    for sk in _skeletons(50, seed=99):
        ids = encode(sk)
        assert " ".join(to_prefix(decode(ids))) == " ".join(to_prefix(sk))
