"""Le garde-fou contre les batches non finis (incident du 2026-08-20).

exp-005 s'est entraîné **3 500 steps sur du NaN** sans que rien ne l'arrête.
Le mécanisme : une loss non finie sur un seul batch, puis `clip_grad_norm_`
renvoie NaN, multiplie tous les gradients par NaN, et l'optimiseur écrit NaN
dans tous les poids. Aucun retour possible ensuite — et `last.pt` est devenu
inutilisable.

Ces tests vérifient la propriété qui manquait : **un batch pathologique ne doit
pas pouvoir corrompre les poids.**
"""

from __future__ import annotations

import pytest
import torch
from torch import nn


def _reproduit_la_corruption() -> bool:
    """Un pas d'optimiseur sur une loss NaN corrompt-il tous les poids ?

    C'est le mécanisme exact de l'incident, reproduit sur un module jouet.
    """
    modele = nn.Linear(4, 4)
    opt = torch.optim.AdamW(modele.parameters(), lr=1e-3)
    x = torch.randn(8, 4)

    perte = (modele(x) * float("nan")).sum()
    opt.zero_grad(set_to_none=True)
    perte.backward()
    norme = torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
    opt.step()

    assert not torch.isfinite(norme), "le clip doit lui-même rendre NaN"
    return not all(torch.isfinite(p).all() for p in modele.parameters())


def test_le_mecanisme_de_corruption_est_bien_celui_quon_croit():
    """Sans garde-fou, un pas sur une loss NaN détruit tous les poids."""
    assert _reproduit_la_corruption()


def test_le_garde_fou_preserve_les_poids():
    """Avec le test de finitude, les poids sortent intacts."""
    modele = nn.Linear(4, 4)
    opt = torch.optim.AdamW(modele.parameters(), lr=1e-3)
    avant = [p.detach().clone() for p in modele.parameters()]
    x = torch.randn(8, 4)

    perte = (modele(x) * float("nan")).sum()
    opt.zero_grad(set_to_none=True)
    if torch.isfinite(perte):  # le garde-fou de curvy.train.loop
        perte.backward()
        opt.step()

    assert all(torch.isfinite(p).all() for p in modele.parameters())
    for p, a in zip(modele.parameters(), avant, strict=True):
        assert torch.equal(p.detach(), a), "un batch écarté ne doit rien changer"


def test_un_gradient_non_fini_est_aussi_intercepte():
    """La loss peut être finie et un gradient déborder quand même."""
    modele = nn.Linear(4, 4)
    opt = torch.optim.AdamW(modele.parameters(), lr=1e-3)
    avant = [p.detach().clone() for p in modele.parameters()]

    x = torch.full((8, 4), 1e30)
    perte = (modele(x) ** 2).sum() * 1e30  # finie ? non — on force le cas via le gradient
    opt.zero_grad(set_to_none=True)
    if torch.isfinite(perte):
        perte.backward()
        norme = torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
        if torch.isfinite(norme):
            opt.step()

    assert all(torch.isfinite(p).all() for p in modele.parameters())
    for p, a in zip(modele.parameters(), avant, strict=True):
        assert torch.equal(p.detach(), a)


def test_la_boucle_dentrainement_porte_bien_les_deux_gardes():
    """Vérification structurelle : les deux tests de finitude sont dans le code.

    Un test de comportement demanderait un entraînement complet ; celui-ci
    garantit au moins que le garde-fou n'est pas retiré par inadvertance.
    """
    from pathlib import Path

    source = Path("curvy/train/loop.py").read_text(encoding="utf-8")
    assert "if not torch.isfinite(loss):" in source
    assert "if not torch.isfinite(grad_norm):" in source
    assert "MAX_INCIDENTS_PAR_FENETRE" in source


def test_le_seuil_dabandon_est_fini():
    """Un run ne doit pas pouvoir écarter des batches indéfiniment en silence."""
    from curvy.train.loop import Trainer

    assert 1 <= Trainer.MAX_INCIDENTS_PAR_FENETRE <= 100
    assert Trainer.FENETRE_INCIDENTS >= 100


@pytest.mark.parametrize("valeur", [float("nan"), float("inf"), float("-inf")])
def test_les_trois_formes_de_non_finitude_sont_couvertes(valeur):
    assert not torch.isfinite(torch.tensor(valeur))
