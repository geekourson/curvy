# Curvy

**[English](#english) · [Français](#français)**

---

<a id="english"></a>

## English

Compact symbolic regression. Give it a **2D point cloud** — a curve drawn
freehand, or a set of measurements — and it returns the **mathematical formulas**
that describe it, ranked on a simplicity/accuracy Pareto front.

31.6 M parameters, five Python dependencies, one consumer GPU.

### The idea in one sentence

The model **never predicts a numeric value**. It predicts a *skeleton*, such as
`C * sin(C * x) + C`, and the constants are fitted afterwards by numerical
optimisation.

Guessing *the shape* is a language problem; finding *the numbers* is an
optimisation problem. Two jobs, two tools.

```
point cloud ──▶ affine normalisation ──▶ transformer encoder (a set of points)
                                                  │
                                             cross-attention
                                                  ▼
                                        autoregressive decoder
                                                  │
                                    beam search under an arity mask
                                                  ▼
                                          N candidate skeletons
                                                  │
                                   constant fitting (scipy least squares)
                                                  ▼
                                   Pareto front (complexity, R²)
                                                  ▼
                                          3 to 5 formulas
```

### Measured results

On **1 960 formulas explicitly withheld from training**, plus 29 hand-written
ones. The score is the share of curves recovered at R² ≥ 0.99, judged against
the exact curve on held-out points, never against the noisy samples.

| | oracle | polynomial | **Curvy** |
|---|---|---|---|
| interpolation | 0.791 | 0.669 | **0.684** |
| **extrapolation** | 0.360 | 0.069 | **0.187** |

*(beam 8; sampling noise is ±2.2 points at 95 % on n = 1960)*

The "oracle" is the same measurement given the true skeleton: the score of
someone handed the answer. It is not 100 % because the samples are noisy and
constant fitting sometimes fails.

**In interpolation, `np.polyfit` is indistinguishable from this model** — within
the error margin, in one millisecond, without a GPU. The project does not earn
its keep there. It earns it outside the observed window, where a polynomial
diverges and a formula stays true: **2.7× better**, and on the 29 hand-written
formulas the polynomial scores **zero on all 29** in extrapolation.

On the Runge function `1/(1+25x²)`, the textbook counter-example to polynomial
interpolation: **Curvy recovers it six times out of six, `np.polyfit` zero times
out of six.**

### What it cannot do

- **Curves that fold back.** Circle, heart, loop. The system predicts `y = f(x)`;
  such a curve has two `y` for one `x`. Structural, not a training issue.
- **Discontinuities.** Step, floor, sawtooth: **0.000**. No discontinuous
  operator in the vocabulary.
- **Fast oscillations.** Constant fitting loses the frequency beyond roughly two
  oscillations per unit width. The model can propose the right formula and still
  fail, because the holes cannot be filled.
- **Deep formulas.** Past eight tree levels a polynomial takes the lead, and a
  137-character expression is no more readable than eight decimal coefficients.

### Getting started

```bash
uv venv --python 3.12 .venv
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.1 numpy scipy sympy matplotlib
make setup     # editable install + pytest and ruff
make env       # what compute the machine actually offers
make test      # 230 tests
```

Heavy artefacts (venv, datasets, checkpoints) live outside the repository, under
`CURVY_DATA_ROOT`. Set it before anything else:

```bash
export CURVY_DATA_ROOT=~/curvy-data
```

`make help` lists every target.

#### Reproducing the model

```bash
# 1. Skeletons. The only symbolic step, paid once. ~75 s on 12 cores.
make data N=2000000 SEED=42

# 2. Freeze the test set BEFORE training, or the final number means nothing.
#    The script refuses to overwrite an existing set and prints a SHA-256 that
#    every published measurement must cite. See benchmarks/testset-v1.json.
.venv/bin/python scripts/build_testset.py

# 3. Train. ~6 h for 40 000 steps on an RTX 3090.
./scripts/launch_run.sh exp-005 v1 40000
CURVY_REPRENDRE=1 ./scripts/launch_run.sh exp-005 v1 40000   # resume

# 4. Evaluate on the frozen set. The SHA-256 is checked on every run.
.venv/bin/python scripts/eval_testset.py --run exp-005 --preset v1 --beam 8

# 5. Serve the browser demo.
make demo      # http://127.0.0.1:8001
```

The launcher excludes the test-set skeletons by default, runs inside `tmux`, and
caps the CPU footprint at five workers at low priority. That last point is not
comfort: an eight-worker run once took the machine's network down. A checkpoint
is written every 1 000 steps together with the optimiser state, so handing the
GPU back to someone else costs at most 1 000 steps, never the run.

#### Before training on a shared machine

The GPU allow-list is set **by UUID**, not by index: a UUID survives
renumbering, an index does not. Put yours — `nvidia-smi -L` prints it — in the
`CURVY_CUDA_ALLOW` variable of the `Makefile` and of `scripts/launch_run.sh`.
An allow-list that matches nothing **raises** rather than silently falling back
to the CPU.

Worth knowing: the CUDA runtime numbers GPUs by compute capability while
`nvidia-smi` numbers them by PCI bus. `cuda:0` and `nvidia-smi`'s `0` need not
be the same card.

### Layout

| | |
|---|---|
| `curvy/data/` | grammar, sampling, canonicalisation, point clouds, dataset |
| `curvy/model/` | the transformer |
| `curvy/train/` | training loop, metrics, checkpoints |
| `curvy/infer/` | beam search, key/value cache, constant fitting, Pareto front |
| `curvy/serve/` | HTTP service, rate limiter, drawing pipeline |
| `scripts/` | generation, evaluation, figures, launchers |
| `tests/` | 230 tests |

The HTTP service deliberately uses the standard library rather than a web
framework: five dependencies is a feature, and a service that does one thing
does not need a router.

### License

MIT. See [LICENSE](LICENSE).

---

<a id="français"></a>

## Français

Régression symbolique compacte. On lui donne un **nuage de points 2D**, une
courbe tracée à main levée ou un jeu de mesures, et il rend les **formules
mathématiques** qui le décrivent, classées sur un front de Pareto
simplicité / précision.

31,6 M de paramètres, cinq dépendances Python, une carte grand public.

### L'idée en une phrase

Le modèle ne prédit **jamais** de valeur numérique. Il prédit un *squelette*,
`C * sin(C * x) + C`, et les constantes sont ajustées après coup par
optimisation numérique.

Deviner *la forme* est un problème de langage, trouver *les nombres* est un
problème d'optimisation. Deux métiers, deux outils.

```
nuage de points ──▶ normalisation affine ──▶ encodeur transformer (un ensemble)
                                                       │
                                                 attention croisée
                                                       ▼
                                             décodeur autorégressif
                                                       │
                                     beam search sous masque d'arité
                                                       ▼
                                            N squelettes candidats
                                                       │
                                   ajustement des constantes (scipy)
                                                       ▼
                                    front de Pareto (complexité, R²)
                                                       ▼
                                              3 à 5 formules
```

### Résultats mesurés

Sur **1 960 formules explicitement retirées de l'entraînement**, plus 29 écrites
à la main. Le score est la part de courbes retrouvées à R² ≥ 0,99, jugé contre
la courbe exacte sur des points tenus à l'écart, jamais contre les points
bruités.

| | oracle | polynôme | **Curvy** |
|---|---|---|---|
| interpolation | 0,791 | 0,669 | **0,684** |
| **extrapolation** | 0,360 | 0,069 | **0,187** |

*(beam 8 ; le bruit d'échantillonnage vaut ±2,2 points à 95 % sur n = 1960)*

L'« oracle » est la même mesure appliquée au vrai squelette : le score de
quelqu'un à qui on aurait donné la solution. Il ne vaut pas 100 % parce que les
points sont bruités et que l'ajustement des constantes échoue parfois.

**En interpolation, `np.polyfit` est indiscernable de ce modèle**, à l'intérieur
de la marge d'erreur, en une milliseconde et sans GPU. Le projet ne se justifie
pas là. Il se justifie hors de la fenêtre observée, où un polynôme diverge et où
une formule reste vraie : **2,7 fois mieux**, et sur les 29 formules écrites à
la main le polynôme rend **zéro sur 29** en extrapolation.

Sur la fonction de Runge `1/(1+25x²)`, le contre-exemple classique de
l'interpolation polynomiale : **Curvy la retrouve six fois sur six,
`np.polyfit` zéro fois sur six.**

### Ce qu'il ne sait pas faire

- **Les courbes qui reviennent en arrière.** Cercle, cœur, boucle. Le système
  prédit `y = f(x)` ; une telle courbe a deux `y` pour un même `x`. C'est
  structurel, pas un défaut d'entraînement.
- **Les discontinuités.** Marche, plancher, dent de scie : **0,000**. Aucun
  opérateur discontinu dans le vocabulaire.
- **Les oscillations rapides.** L'ajustement perd la fréquence au-delà d'environ
  deux oscillations par unité de largeur. Le modèle peut proposer la bonne
  formule et échouer quand même, faute de pouvoir remplir les trous.
- **Les formules profondes.** Au-delà de huit niveaux d'arbre, le polynôme
  reprend l'avantage, et une expression de 137 caractères n'est pas plus lisible
  que huit coefficients décimaux.

### Démarrage

```bash
uv venv --python 3.12 .venv
uv pip install --index-url https://download.pytorch.org/whl/cu128 \
    torch==2.9.1 numpy scipy sympy matplotlib
make setup     # installation en editable + pytest et ruff
make env       # ce que la machine offre réellement comme calcul
make test      # 230 tests
```

Les artefacts lourds (venv, jeux de données, points de reprise) vivent hors du
dépôt, sous `CURVY_DATA_ROOT`. À définir avant toute chose :

```bash
export CURVY_DATA_ROOT=~/curvy-data
```

`make help` liste toutes les cibles.

#### Reproduire le modèle

```bash
# 1. Les squelettes. Seule étape symbolique, payée une fois. ~75 s sur 12 cœurs.
make data N=2000000 SEED=42

# 2. Geler le jeu de test AVANT d'entraîner, sinon le chiffre final ne vaut
#    rien. Le script refuse d'écraser un jeu existant et publie une empreinte
#    SHA-256 que toute mesure doit citer. Voir benchmarks/testset-v1.json.
.venv/bin/python scripts/build_testset.py

# 3. Entraîner. ~6 h pour 40 000 étapes sur une RTX 3090.
./scripts/launch_run.sh exp-005 v1 40000
CURVY_REPRENDRE=1 ./scripts/launch_run.sh exp-005 v1 40000   # reprendre

# 4. Évaluer sur le jeu figé. L'empreinte est vérifiée à chaque exécution.
.venv/bin/python scripts/eval_testset.py --run exp-005 --preset v1 --beam 8

# 5. Servir la démo navigateur.
make demo      # http://127.0.0.1:8001
```

Le lanceur exclut par défaut les squelettes du jeu de test, tourne dans `tmux`,
et bride l'empreinte processeur à cinq travailleurs en priorité basse. Ce
dernier point n'est pas du confort : un entraînement à huit travailleurs a déjà
fait tomber le réseau de la machine. Un point de reprise est écrit toutes les
1 000 étapes avec l'état de l'optimiseur, si bien que rendre la carte à
quelqu'un d'autre coûte au pire 1 000 étapes, jamais l'entraînement.

#### Avant d'entraîner sur une machine partagée

La liste blanche des GPU se règle **par UUID** et non par index : un UUID survit
à une renumérotation, un index non. Renseignez le vôtre, que `nvidia-smi -L`
affiche, dans la variable `CURVY_CUDA_ALLOW` du `Makefile` et de
`scripts/launch_run.sh`. Une liste qui ne correspond à rien **lève une erreur**
plutôt que de basculer en silence sur le CPU.

À savoir : le runtime CUDA numérote les cartes par puissance quand `nvidia-smi`
les numérote par bus PCI. `cuda:0` et le `0` de `nvidia-smi` ne désignent pas
forcément la même carte.

### Organisation

| | |
|---|---|
| `curvy/data/` | grammaire, tirage, canonicalisation, nuages, jeu de données |
| `curvy/model/` | le transformer |
| `curvy/train/` | boucle d'entraînement, métriques, points de reprise |
| `curvy/infer/` | beam search, cache clé/valeur, ajustement, front de Pareto |
| `curvy/serve/` | service HTTP, limiteur de débit, traitement du tracé |
| `scripts/` | génération, évaluation, figures, lanceurs |
| `tests/` | 230 tests |

Le service HTTP s'appuie délibérément sur la bibliothèque standard plutôt que
sur un cadriciel web : cinq dépendances est une qualité, et un service qui fait
une seule chose n'a pas besoin d'un routeur.

### Licence

MIT. Voir [LICENSE](LICENSE).
