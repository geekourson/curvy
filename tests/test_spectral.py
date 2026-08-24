"""Initialisation spectrale de l'ajustement des constantes (2026-08-20).

L'initialisation log-uniforme cherchait la fréquence au hasard dans
``[0,05 ; 20]`` et décrochait au-delà de 12 rad : avec le squelette **exact**,
``C·sin(30x)+C`` s'ajustait à R² 0,002. Le modèle pouvait proposer la bonne
formule et scorer zéro.

La fréquence se lit pourtant directement dans les points, par transformée de
Fourier. Ces tests figent le gain et l'absence de régression.
"""

from __future__ import annotations

import numpy as np
import pytest

from curvy.data.expr import from_prefix
from curvy.infer.fit import fit_constants, frequences_candidates
from curvy.seeding import make_rng

X = np.linspace(-1.0, 1.0, 200)
SINUS = from_prefix(["add", "mul", "C", "sin", "mul", "C", "x", "C"])


@pytest.mark.parametrize("freq", [3.0, 7.5, 16.0, 25.0, 40.0])
def test_la_fft_retrouve_la_pulsation(freq):
    lues = frequences_candidates(X, np.sin(freq * X))
    assert lues, "aucun pic détecté"
    assert min(abs(v - freq) for v in lues) < 0.15 * freq


@pytest.mark.parametrize("freq", [16.0, 25.0, 30.0, 45.0])
def test_les_hautes_frequences_sajustent_maintenant(freq):
    """Le décrochage au-delà de 12 rad, figé en test."""
    y = np.sin(freq * X)
    avec = fit_constants(SINUS, X, y, make_rng(0), spectral=True)
    assert avec.r2_fit > 0.99, f"{freq} rad devrait s'ajuster"


def test_le_tirage_aleatoire_seul_echoue_bien_au_dela_de_douze():
    """Le contre-exemple : sans l'initialisation spectrale, ça ne marche pas.

    Si ce test se met à échouer, c'est que le repli aléatoire est devenu bon
    tout seul — et l'initialisation spectrale ne servirait plus à rien.
    """
    sans = fit_constants(SINUS, X, np.sin(30.0 * X), make_rng(0), spectral=False)
    assert sans.r2_fit < 0.5


@pytest.mark.parametrize(
    ("prefixe", "f"),
    [
        ("add mul C x C", lambda x: 3.0 * x - 1.0),
        ("add mul C sq x C", lambda x: 2.0 * x**2 - 1.0),
        ("add mul C exp mul C x C", lambda x: 2.0 * np.exp(1.5 * x)),
        ("add mul C tanh mul C x C", lambda x: np.tanh(4.0 * x)),
    ],
)
def test_aucune_regression_sur_les_signaux_non_periodiques(prefixe, f):
    node = from_prefix(prefixe.split())
    y = f(X)
    avant = fit_constants(node, X, y, make_rng(0), spectral=False)
    apres = fit_constants(node, X, y, make_rng(0), spectral=True)
    assert apres.r2_fit >= avant.r2_fit - 1e-9


def test_la_fft_ne_renvoie_rien_sur_un_signal_plat():
    """Pas de pic exploitable : on retombe sur le tirage, sans planter."""
    assert frequences_candidates(X, np.zeros_like(X)) == []


def test_la_fft_supporte_un_echantillonnage_irregulier():
    """Les nuages du projet ne sont jamais à pas régulier."""
    rng = make_rng(4)
    x = np.sort(rng.uniform(-1, 1, 180))
    lues = frequences_candidates(x, np.sin(9.0 * x))
    assert lues and min(abs(v - 9.0) for v in lues) < 1.5


def test_la_fft_refuse_un_signal_trop_court():
    assert frequences_candidates(np.linspace(-1, 1, 5), np.zeros(5)) == []


def test_linitialisation_spectrale_est_plus_rapide_que_le_tirage():
    """Viser converge immédiatement ; tirer brûle six essais pour rien."""
    import time

    y = np.sin(20.0 * X)
    t0 = time.perf_counter()
    fit_constants(SINUS, X, y, make_rng(0), spectral=False)
    sans = time.perf_counter() - t0
    t0 = time.perf_counter()
    fit_constants(SINUS, X, y, make_rng(0), spectral=True)
    avec = time.perf_counter() - t0
    assert avec < sans, f"spectral {1000 * avec:.1f} ms contre {1000 * sans:.1f} ms"
