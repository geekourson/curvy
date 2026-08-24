"""Limitation de débit pour le service public.

Le service est destiné à être mis en ligne avec l'article, sans
authentification. Il n'écrit plus rien sur disque (journal du 2026-08-22), donc
le risque n'est plus la croissance : c'est la **disponibilité**. Une requête
`/api/formules` coûte ~779 ms de GPU ; un robot à dix requêtes par seconde
monopolise la 3060 et la démo ne répond plus à personne.

**Deux niveaux, parce qu'ils protègent contre deux choses différentes.**

- **par adresse** — empêche un client seul de tout prendre ;
- **global** — protège la carte quand la charge vient de partout à la fois. Sans
  lui, cent adresses respectant chacune leur quota suffiraient à saturer le GPU.

**Le seau à jetons** plutôt qu'un compteur par fenêtre : il autorise une petite
rafale — un humain qui dessine trois courbes d'affilée ne doit pas être bloqué —
tout en bornant le débit moyen. Un compteur par fenêtre laisse au contraire
passer deux fois le quota à cheval sur deux fenêtres.

**La mémoire est bornée.** Un dictionnaire indexé par adresse grandit avec le
nombre d'adresses distinctes : c'est exactement la croissance non bornée qu'on
vient de retirer du disque, transposée en RAM. Les entrées inactives sont donc
purgées, et leur nombre est plafonné.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

__all__ = ["Limiteur", "SeauAJetons"]


@dataclass
class SeauAJetons:
    """Un seau qui se remplit à débit constant et se vide à chaque requête."""

    capacite: float
    par_seconde: float
    jetons: float = field(default=0.0)
    #: `None` et non `0.0` : une horloge qui démarre à zéro est indistinguable
    #: d'une sentinelle à zéro, et le temps écoulé n'était jamais crédité au
    #: premier rechargement. Bogue attrapé par les tests avant toute mise en
    #: ligne.
    dernier: float | None = None

    def __post_init__(self) -> None:
        self.jetons = float(self.capacite)

    def consommer(self, maintenant: float, cout: float = 1.0) -> float:
        """Retourne 0 si la requête passe, sinon l'attente en secondes."""
        if self.dernier is None:
            self.dernier = maintenant
        ecoule = max(0.0, maintenant - self.dernier)
        self.dernier = maintenant
        self.jetons = min(self.capacite, self.jetons + ecoule * self.par_seconde)
        if self.jetons >= cout:
            self.jetons -= cout
            return 0.0
        manque = cout - self.jetons
        return manque / self.par_seconde if self.par_seconde > 0 else float("inf")


class Limiteur:
    """Limitation par adresse et globale, avec purge des adresses inactives."""

    def __init__(
        self,
        par_adresse: tuple[float, float] = (5.0, 0.5),
        global_: tuple[float, float] = (20.0, 2.0),
        max_adresses: int = 4096,
        oubli_s: float = 900.0,
    ) -> None:
        """``(capacite, par_seconde)`` pour chacun des deux niveaux.

        Défauts : une rafale de 5 requêtes par adresse puis une toutes les deux
        secondes ; globalement une rafale de 20 puis deux par seconde — le GPU
        en soutient environ 1,3, le reste attend dans le seau plutôt que d'être
        refusé.
        """
        self.par_adresse = par_adresse
        self.global_ = global_
        self.max_adresses = max_adresses
        self.oubli_s = oubli_s
        self._seaux: dict[str, SeauAJetons] = {}
        self._seau_global = SeauAJetons(*global_)
        self._verrou = threading.Lock()

    def _purger(self, maintenant: float) -> None:
        """Oublie les adresses inactives. Sans cela, le dictionnaire grandit
        avec le nombre d'adresses distinctes — la croissance non bornée qu'on
        vient de retirer du disque, transposée en mémoire."""
        morts = [
            ip
            for ip, s in self._seaux.items()
            if s.dernier is not None and maintenant - s.dernier > self.oubli_s
        ]
        for ip in morts:
            del self._seaux[ip]
        if len(self._seaux) > self.max_adresses:
            # Purge d'urgence : on garde les plus récentes.
            recents = sorted(self._seaux.items(), key=lambda kv: -(kv[1].dernier or 0.0))
            self._seaux = dict(recents[: self.max_adresses // 2])

    def verifier(self, adresse: str, maintenant: float, cout: float = 1.0) -> float:
        """0 si la requête passe, sinon l'attente conseillée en secondes.

        Le seau global n'est débité **que si** le seau de l'adresse a accepté :
        sinon un client bloqué continuerait à épuiser le quota commun.
        """
        with self._verrou:
            self._purger(maintenant)
            seau = self._seaux.get(adresse)
            if seau is None:
                seau = self._seaux[adresse] = SeauAJetons(*self.par_adresse)
            attente = seau.consommer(maintenant, cout)
            if attente > 0.0:
                return attente
            attente_globale = self._seau_global.consommer(maintenant, cout)
            if attente_globale > 0.0:
                # On rend le jeton pris à l'adresse : la requête n'a pas eu lieu.
                seau.jetons = min(seau.capacite, seau.jetons + cout)
                return attente_globale
            return 0.0

    @property
    def adresses_suivies(self) -> int:
        return len(self._seaux)
