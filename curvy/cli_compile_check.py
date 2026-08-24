"""``python -m curvy.cli_compile_check`` — ``torch.compile`` est-il utilisable ici ?

La spec demande de valider ``torch.compile`` sur un module jouet avant de
l'intégrer au training loop. Un module jouet à **forme fixe** ne prouve
pourtant pas grand-chose : notre encodeur reçoit des nuages de 20 à 200 points,
donc des formes **variables**. Le mode de défaillance réaliste n'est pas
« ça plante », c'est « ça recompile à chaque nouvelle taille » et l'entraînement
devient plus lent qu'en eager, silencieusement.

Ce script teste donc trois choses :
1. compile fonctionne et donne le même résultat qu'en eager (tolérance notée) ;
2. combien de recompilations une forme variable déclenche ;
3. le gain réel en débit, forme fixe, mesuré et non supposé.
"""

from __future__ import annotations

import time

import torch
from torch import nn

from curvy.devices import pick_device
from curvy.precision import configure_precision, precision_report
from curvy.seeding import seed_everything


class Toy(nn.Module):
    """Bloc jouet représentatif : projection + attention SDPA + MLP."""

    def __init__(self, d: int = 256, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=-1)
        shape = (b, n, self.heads, d // self.heads)
        q, k, v = (t.view(shape).transpose(1, 2) for t in (q, k, v))
        # SDPA : attention fusionnée de PyTorch, pas de réimplémentation manuelle.
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x + self.proj(a.transpose(1, 2).reshape(b, n, d))
        return x + self.mlp(x)


def _bench(fn, x: torch.Tensor, device: torch.device, iters: int = 30) -> float:
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    for _ in range(5):
        fn(x)
    sync()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    sync()
    return (time.perf_counter() - t0) / iters


def main() -> int:
    seed_everything(42)
    configure_precision(tf32=True)
    info = pick_device("auto")
    device = info.device
    print("device    :", info)
    print("précision :", precision_report(device))
    print()

    model = Toy().to(device).eval()
    x = torch.randn(32, 128, 256, device=device)

    with torch.no_grad():
        ref = model(x)

    print("--- 1. compile fonctionne-t-il, et donne-t-il le même résultat ? ---")
    t_compile0 = time.perf_counter()
    compiled = torch.compile(model)
    with torch.no_grad():
        got = compiled(x)
    warm = time.perf_counter() - t_compile0
    diff = (ref - got).abs().max().item()
    print(f"première passe (compilation incluse) : {warm:.2f} s")
    print(f"écart max eager vs compiled          : {diff:.3e}")
    print(f"verdict                              : {'OK' if diff < 1e-3 else 'ÉCART SUSPECT'}")
    print()

    print("--- 2. formes variables : combien de recompilations ? ---")
    counter = {"n": 0}
    try:
        import torch._dynamo as dynamo

        dynamo.reset()
        compiled_dyn = torch.compile(model)
        sizes = [20, 47, 96, 128, 200, 47, 128]
        t0 = time.perf_counter()
        with torch.no_grad():
            for n in sizes:
                compiled_dyn(torch.randn(8, n, 256, device=device))
        dt = time.perf_counter() - t0
        stats = dynamo.utils.counters.get("stats", {})
        counter["n"] = stats.get("unique_graphs", -1)
        print(f"tailles testées : {sizes}")
        print(f"graphes uniques compilés : {counter['n']}")
        print(f"temps total : {dt:.2f} s")
    except Exception as exc:  # noqa: BLE001
        print(f"instrumentation dynamo indisponible : {type(exc).__name__}: {exc}")
    print()

    print("--- 3. gain réel à forme fixe ---")
    with torch.no_grad():
        t_eager = _bench(model, x, device)
        t_comp = _bench(compiled, x, device)
    print(f"eager    : {t_eager * 1e3:.3f} ms/iter")
    print(f"compiled : {t_comp * 1e3:.3f} ms/iter")
    print(f"gain     : x{t_eager / t_comp:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
