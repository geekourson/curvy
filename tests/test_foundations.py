"""Tests de fondation : l'environnement et les briques transverses.

Volontairement modestes — leur rôle est de prouver que la chaîne
lint/test/import tourne, et de verrouiller deux invariants qui nous ont déjà
coûté du temps (cf. journal du 2026-08-19).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from curvy.config import DATA_ROOT, REPO_ROOT
from curvy.devices import describe_backends, pick_device
from curvy.seeding import DEFAULT_SEED, make_rng, seed_everything


def _exiger_gpu_libre() -> None:
    """Passe le test si le GPU autorisé est occupé par un autre travail.

    Sans ce garde-fou, la suite échoue dès qu'un entraînement ou une évaluation
    tourne — un test rouge qui ne signale aucune régression est pire qu'un test
    absent : il apprend à ignorer le rouge.
    """
    from curvy.devices import pick_device

    try:
        pick_device("auto")
    except RuntimeError as exc:
        pytest.skip(f"GPU autorisé occupé : {exc}")


def test_package_importable():
    import curvy

    assert curvy.__version__


def test_paths_are_absolute():
    assert REPO_ROOT.is_absolute()
    assert DATA_ROOT.is_absolute()


def test_seed_everything_is_reproducible():
    """Deux semis identiques doivent produire exactement les mêmes tirages."""
    rng_a = seed_everything(DEFAULT_SEED)
    a = (torch.randn(16).tolist(), rng_a.random(16).tolist())
    rng_b = seed_everything(DEFAULT_SEED)
    b = (torch.randn(16).tolist(), rng_b.random(16).tolist())
    assert a == b


def test_make_rng_is_independent_of_global_state():
    """Le Generator ne doit pas dépendre de l'état global numpy."""
    a = make_rng(7).random(8).tolist()
    np.random.seed(1234)  # noqa: NPY002
    b = make_rng(7).random(8).tolist()
    assert a == b


def test_seed_everything_differs_across_seeds():
    seed_everything(1)
    a = torch.randn(16).tolist()
    seed_everything(2)
    assert torch.randn(16).tolist() != a


def test_pick_device_cpu_is_always_available():
    info = pick_device("cpu")
    assert info.device.type == "cpu"


def test_pick_device_auto_returns_usable_device():
    _exiger_gpu_libre()
    info = pick_device("auto")
    x = torch.zeros(4, device=info.device)  # doit allouer sans exception
    assert x.sum().item() == 0.0


def test_describe_backends_has_expected_keys():
    info = describe_backends()
    for key in ("torch", "cuda_available", "mps_available", "mps_built"):
        assert key in info


@pytest.mark.gpu
def test_probing_a_dead_gpu_does_not_leave_current_device_broken():
    """Régression : une sonde mémoire ratée laissait ``current_device`` sur le
    GPU mort, et le premier ``synchronize()`` sans argument explosait."""
    if not torch.cuda.is_available():
        pytest.skip("pas de CUDA")
    _exiger_gpu_libre()
    info = pick_device("auto")
    if info.device.type != "cuda":
        pytest.skip("aucun GPU CUDA exploitable")
    assert torch.cuda.current_device() == info.device.index
    torch.randn(8, device="cuda")
    torch.cuda.synchronize()  # sans argument : c'est là que ça cassait


# --- Politique d'allocation GPU ----------------------------------


def test_uuid_normalisation_handles_both_spellings():
    """Régression : nvidia-smi préfixe l'UUID par ``GPU-``, torch non.

    L'oubli de ce détail vidait silencieusement l'allowlist, et le projet se
    repliait sur le CPU sans rien dire.
    """
    from curvy.devices import _normalise_uuid

    smi = "GPU-1234ABCD-0000-0000-0000-000000000000"
    torch_side = "1234abcd-0000-0000-0000-000000000000"
    assert _normalise_uuid(smi) == _normalise_uuid(torch_side)


def test_allowlist_parses_indices_and_uuids(monkeypatch):
    from curvy.devices import _allowlist

    monkeypatch.delenv("CURVY_CUDA_ALLOW", raising=False)
    assert _allowlist() is None

    monkeypatch.setenv("CURVY_CUDA_ALLOW", "1, GPU-ABC-def ")
    assert _allowlist() == {"1", "abc-def"}


def test_allowlist_matching_nothing_raises_instead_of_falling_back_to_cpu(monkeypatch):
    """Un repli silencieux sur CPU ferait « tourner » un entraînement 100x trop lent."""
    if not torch.cuda.is_available():
        pytest.skip("pas de CUDA")
    _exiger_gpu_libre()
    monkeypatch.setenv("CURVY_CUDA_ALLOW", "GPU-0000-inexistant")
    with pytest.raises(RuntimeError, match="Aucun GPU autorisé"):
        pick_device("auto")


# --- Précision numérique -----------------------------------------------------


def test_precision_report_works_on_cpu():
    """Régression : le rapport interrogeait CUDA même pour un device CPU."""
    from curvy.precision import precision_report

    rep = precision_report(torch.device("cpu"))
    assert rep["device"] == "cpu"
    assert "bf16_supported" in rep


def test_configure_precision_is_idempotent_and_safe_without_cuda():
    from curvy.precision import configure_precision

    configure_precision(tf32=True)
    configure_precision(tf32=True)
    assert torch.get_float32_matmul_precision() in ("high", "highest")


@pytest.mark.gpu
def test_bf16_is_supported_on_ampere():
    from curvy.precision import bf16_supported

    if not torch.cuda.is_available():
        pytest.skip("pas de CUDA")
    _exiger_gpu_libre()
    info = pick_device("auto")
    if info.device.type != "cuda":
        pytest.skip("aucun GPU CUDA exploitable")
    assert bf16_supported(info.device), "bf16 attendu sur Ampere (sm_86)"
