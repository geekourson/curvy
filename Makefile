# Curvy — cibles principales.
# Le venv et les artefacts lourds vivent hors du dépôt, voir CURVY_DATA_ROOT.
PY      := .venv/bin/python
SEED    ?= 42
RUN     ?= exp-005
PRESET  ?= v1
# 127.0.0.1 par défaut. `make demo HOST=0.0.0.0` pour ouvrir au réseau local.
HOST    ?= 127.0.0.1
PORT    ?= 8001
BEAM    ?= 48
DEVICE  ?= auto
N       ?= 500000

# Le runtime CUDA numérote les GPU par puissance et non par bus PCI ; on force
# l'ordre PCI pour que `cuda:0` désigne la même carte que dans nvidia-smi.
export CUDA_DEVICE_ORDER := PCI_BUS_ID

# Liste blanche des GPU utilisables, par UUID plutôt que par index : un UUID
# survit à une renumérotation, un index non. À renseigner avec le vôtre, que
# `nvidia-smi -L` affiche. Une liste qui ne correspond à rien lève une erreur
# plutôt que de basculer en silence sur le CPU.
export CURVY_CUDA_ALLOW := GPU-1234abcd-0000-0000-0000-000000000000

# Vérifie la VRAM libre et signale les squatteurs avant tout entraînement.
gpu-check:       ## État de la 3090 + processus qui l'occupent
	$(PY) -m curvy.cli_gpu

.PHONY: help setup env gpu-check data train train-tmux curves eval demo test lint fmt clean

help:            ## Liste les cibles
	@grep -E '^[a-z-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

setup:           ## Installe le paquet en editable + outils de dev
	uv pip install --python $(PY) -e ".[dev]"

env:             ## Affiche l'environnement de calcul retenu (livrable Phase 0)
	$(PY) -m curvy.cli_env

data:            ## Génère l'ensemble de squelettes (Phase 1)
	$(PY) -m curvy.data.generate --n $(N) --seed $(SEED)

train: gpu-check ## Entraîne un modèle (Phase 4) — refuse de démarrer si la VRAM est prise
	$(PY) -m curvy.train.run --seed $(SEED)

train-tmux: gpu-check ## Lance les entraînements longs dans tmux (survit à la déconnexion)
	./scripts/launch_training.sh

curves:          ## Trace les courbes d'un run : make curves RUN=exp-001
	$(PY) scripts/plot_training.py --run $(RUN)

eval:            ## Évalue sur le jeu de test figé (Phase 6)
	$(PY) -m curvy.eval.run --device $(DEVICE)

demo:            ## Sert la démo navigateur sur la 3060 (Phase 8) — http://127.0.0.1:8001
	$(PY) -m curvy.serve.app --run $(RUN) --preset $(PRESET) --host $(HOST) --port $(PORT) --beam $(BEAM)

test:            ## Tests rapides
	$(PY) -m pytest -m "not slow"

lint:            ## Ruff (lint + format check)
	$(PY) -m ruff check curvy tests scripts
	$(PY) -m ruff format --check curvy tests scripts

fmt:             ## Ruff format
	$(PY) -m ruff format curvy tests scripts

clean:           ## Supprime les caches Python
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
