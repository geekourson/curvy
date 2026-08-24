"""Limitation de débit (2026-08-22).

Le service part en ligne avec l'article, sans authentification. Il n'écrit plus
sur disque, donc le risque n'est plus la croissance mais la **disponibilité** :
une requête coûte ~779 ms de GPU.

Ces tests portent sur les propriétés qui décident si la protection tient : la
rafale autorisée, le débit moyen, le fait qu'un client bloqué n'épuise pas le
quota commun, et **la mémoire bornée** — un dictionnaire par adresse est la même
croissance non bornée qu'on vient de retirer du disque, transposée en RAM.
"""

from __future__ import annotations

from curvy.serve.debit import Limiteur, SeauAJetons


def test_le_seau_autorise_une_rafale_puis_ralentit():
    """Un humain qui dessine trois courbes d'affilée ne doit pas être bloqué."""
    seau = SeauAJetons(capacite=5.0, par_seconde=0.5)
    for _ in range(5):
        assert seau.consommer(0.0) == 0.0
    assert seau.consommer(0.0) > 0.0, "la sixième doit attendre"


def test_le_seau_se_remplit_avec_le_temps():
    seau = SeauAJetons(capacite=5.0, par_seconde=0.5)
    for _ in range(5):
        seau.consommer(0.0)
    assert seau.consommer(1.0) > 0.0, "après 1 s, un demi-jeton seulement"
    assert seau.consommer(2.0) == 0.0, "après 2 s, un jeton entier"


def test_lattente_annoncee_est_suffisante():
    """Le `Retry-After` doit être tenable : attendre ce délai doit débloquer."""
    seau = SeauAJetons(capacite=2.0, par_seconde=0.5)
    seau.consommer(0.0)
    seau.consommer(0.0)
    attente = seau.consommer(0.0)
    assert attente > 0.0
    assert seau.consommer(attente) == 0.0


def test_le_seau_ne_deborde_pas():
    """Une longue inactivité ne doit pas offrir une rafale illimitée."""
    seau = SeauAJetons(capacite=3.0, par_seconde=1.0)
    seau.consommer(0.0)
    seau.consommer(10_000.0)  # très longue pause
    assert seau.jetons <= 3.0


def test_une_adresse_bloquee_nepuise_pas_le_quota_commun():
    """Sinon un seul robot priverait tout le monde en étant lui-même refusé."""
    lim = Limiteur(par_adresse=(2.0, 0.1), global_=(10.0, 1.0))
    for _ in range(2):
        assert lim.verifier("1.1.1.1", 0.0) == 0.0
    for _ in range(20):
        lim.verifier("1.1.1.1", 0.0)  # toutes refusées
    # Le quota global doit être intact pour les autres. Chaque adresse ne peut
    # en prendre que sa propre capacité — d'où quatre adresses distinctes pour
    # vérifier qu'il reste bien huit jetons communs.
    passees = sum(1 for i in range(4) for _ in range(2) if lim.verifier(f"3.3.3.{i}", 0.0) == 0.0)
    assert passees == 8, f"{passees} requêtes passées au lieu de 8"


def test_le_quota_global_protege_quand_la_charge_vient_de_partout():
    """Cent adresses respectant chacune leur quota satureraient le GPU."""
    lim = Limiteur(par_adresse=(5.0, 0.5), global_=(10.0, 1.0))
    acceptees = sum(1 for i in range(100) if lim.verifier(f"10.0.0.{i}", 0.0) == 0.0)
    assert acceptees == 10, f"{acceptees} acceptées au lieu de 10"


def test_les_adresses_inactives_sont_oubliees():
    """La mémoire est bornée : c'est la croissance non bornée retirée du disque
    qui reviendrait par la RAM."""
    lim = Limiteur(oubli_s=100.0)
    for i in range(50):
        lim.verifier(f"10.0.0.{i}", 0.0)
    assert lim.adresses_suivies == 50
    lim.verifier("10.0.1.1", 1000.0)  # bien après l'oubli
    assert lim.adresses_suivies == 1


def test_le_nombre_dadresses_suivies_est_plafonne():
    """Même sans pause, un balayage d'adresses ne doit pas gonfler la mémoire."""
    lim = Limiteur(max_adresses=64, oubli_s=1e9)
    for i in range(500):
        lim.verifier(f"10.{i // 256}.{i % 256}.1", float(i))
    assert lim.adresses_suivies <= 64


def test_deux_adresses_ont_des_quotas_independants():
    lim = Limiteur(par_adresse=(2.0, 0.1), global_=(100.0, 10.0))
    assert lim.verifier("1.1.1.1", 0.0) == 0.0
    assert lim.verifier("1.1.1.1", 0.0) == 0.0
    assert lim.verifier("1.1.1.1", 0.0) > 0.0
    assert lim.verifier("2.2.2.2", 0.0) == 0.0, "l'autre adresse ne doit pas payer"


def test_un_cout_plus_eleve_consomme_davantage():
    """Une requête à 779 ms de GPU ne vaut pas un rééchantillonnage à 4 ms."""
    lim = Limiteur(par_adresse=(4.0, 0.1), global_=(100.0, 10.0))
    assert lim.verifier("1.1.1.1", 0.0, cout=4.0) == 0.0
    assert lim.verifier("1.1.1.1", 0.0, cout=1.0) > 0.0


# --- identification du client derrière un proxy (2026-08-24) ---


def test_xff_ignore_sans_proxy_declare():
    """`X-Forwarded-For` est fourni par le client : le croire sans condition
    permettrait de s'inventer une adresse par requête et de contourner le
    quota."""
    from curvy.serve.app import Etat, Handler

    Etat.proxys = frozenset()
    h = object.__new__(Handler)
    h.client_address = ("203.0.113.9", 5000)
    h.headers = {"X-Forwarded-For": "1.2.3.4"}
    assert Handler._adresse(h) == "203.0.113.9"


def test_xff_honore_depuis_un_proxy_declare():
    from curvy.serve.app import Etat, Handler

    Etat.proxys = frozenset({"127.0.0.1"})
    h = object.__new__(Handler)
    h.client_address = ("127.0.0.1", 5000)
    h.headers = {"X-Forwarded-For": "198.51.100.7"}
    assert Handler._adresse(h) == "198.51.100.7"
    Etat.proxys = frozenset()


def test_xff_prend_le_dernier_maillon():
    """Le client peut préfixer la chaîne de fausses adresses ; seule la
    dernière a été ajoutée par le proxy de confiance."""
    from curvy.serve.app import Etat, Handler

    Etat.proxys = frozenset({"127.0.0.1"})
    h = object.__new__(Handler)
    h.client_address = ("127.0.0.1", 5000)
    h.headers = {"X-Forwarded-For": "9.9.9.9, 8.8.8.8, 198.51.100.7"}
    assert Handler._adresse(h) == "198.51.100.7"
    Etat.proxys = frozenset()
