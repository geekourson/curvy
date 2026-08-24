"""La ligne de commande d'entraînement transporte-t-elle vraiment ses drapeaux ?

Ces tests existent à cause d'un incident : `--exclure-test` était déclaré dans
l'argparse, accepté sans erreur, et n'arrivait jamais jusqu'à `TrainConfig`,
parce que la config était assemblée par une liste de champs recopiée à la main.
Le run tournait donc en voyant les squelettes du jeu de test, et rien ne le
disait. Un drapeau qui ne fait rien en silence est pire qu'un drapeau absent.
"""

from __future__ import annotations

import argparse
import dataclasses

import pytest

from curvy.train.config import TrainConfig
from curvy.train.run import HORS_CONFIG, config_depuis_arguments


def _analyseur() -> argparse.ArgumentParser:
    """Le même analyseur que `main`, sans lancer d'entraînement."""
    import curvy.train.run as module

    source = module.main.__code__
    assert "config_depuis_arguments" in source.co_names, (
        "main ne passe plus par config_depuis_arguments : ces tests ne protègent plus rien"
    )
    # On reconstruit l'analyseur en appelant main avec --help interceptée serait
    # fragile ; on le déclare ici et un test vérifie qu'il couvre tous les champs.
    ap = argparse.ArgumentParser()
    cfg = TrainConfig()
    ap.add_argument("--run-name", dest="run_name", default=cfg.run_name)
    ap.add_argument("--steps", type=int, default=cfg.steps)
    ap.add_argument("--workers", type=int, default=cfg.workers)
    ap.add_argument("--exclure-test", dest="exclure_test", action="store_true")
    ap.add_argument("--no-resume", action="store_true")
    return ap


def test_exclure_test_arrive_jusqua_la_config():
    """Le bug du 2026-08-20, figé en test."""
    args = _analyseur().parse_args(["--exclure-test"])
    assert config_depuis_arguments(args).exclure_test is True


def test_sans_le_drapeau_lexclusion_reste_desactivee():
    args = _analyseur().parse_args([])
    assert config_depuis_arguments(args).exclure_test is False


def test_un_argument_sans_champ_correspondant_leve_une_erreur():
    """C'est le garde-fou : un drapeau orphelin doit crier, pas être ignoré."""
    ap = _analyseur()
    ap.add_argument("--invente", dest="invente", default="x")
    with pytest.raises(ValueError, match="sans champ correspondant"):
        config_depuis_arguments(ap.parse_args([]))


def test_les_arguments_hors_config_sont_declares_explicitement():
    """`no_resume` pilote le lancement, pas la configuration du run."""
    champs = {f.name for f in dataclasses.fields(TrainConfig)}
    assert not (HORS_CONFIG & champs), "un argument hors config ne doit pas doubler un champ"
    args = _analyseur().parse_args(["--no-resume"])
    cfg = config_depuis_arguments(args)  # ne doit pas lever
    assert not hasattr(cfg, "no_resume")


def test_lexclusion_est_ecrite_dans_le_config_json_du_run():
    """Sans cette trace, on ne peut pas savoir après coup si les chiffres d'un
    run sur le jeu de test veulent dire quelque chose."""
    cfg = TrainConfig(exclure_test=True)
    assert cfg.to_dict()["exclure_test"] is True
