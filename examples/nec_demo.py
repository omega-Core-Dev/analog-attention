"""Demo do NEC — Núcleo Emocional-Cognitivo.

Dois ciclos com âncora de contexto — a entidade mantém
o fio do estado entre interações.
"""
from analog_attention.nec import processar_nec

print("=" * 60)
print("  NEC · Núcleo Emocional-Cognitivo · v0.1")
print("  Baseado em experiência direta humana · QE-05")
print("=" * 60)

# Ciclo 1
print("\n>>> CICLO 1")
c1 = processar_nec(
    "quero entender como a atenção funciona nesse sistema novo",
    verbose=True,
)

# Ciclo 2 — contexto diferente, âncora ativa
print("\n>>> CICLO 2  (âncora do ciclo 1 ativa)")
c2 = processar_nec(
    "a frequência define quem responde e quem fica em silêncio",
    ancora=c1.ancora,
    verbose=True,
)

# Ciclo 3 — salto intencional
print("\n>>> CICLO 3  (salto de contexto)")
c3 = processar_nec(
    "urgente preciso de ajuda imediata agora",
    ancora=c2.ancora,
    verbose=True,
)

# Resumo
print("\n" + "=" * 60)
print("  RESUMO DOS CICLOS")
print("=" * 60)
for i, c in enumerate([c1, c2, c3], 1):
    salto = "⚡ SALTO" if c.ea.flag_salto else "→ contínuo"
    print(
        f"  Ciclo {i}: {c.resposta.modo:<14} "
        f"EE(val={c.ee.valencia:+.2f} int={c.ee.intensidade:.2f}) "
        f"RA(Δ={c.ra.delta_novidade:.2f}) {salto}"
    )
