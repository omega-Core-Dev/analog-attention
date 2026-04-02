"""Protótipo de entrada humana — dois ciclos com âncora de contexto.

Demonstra o pipeline completo:
    texto → tokenizer → broadcast parabólico → cadeia harmônica → frequency log
"""
from analog_attention import processar

# ── Ciclo 1 ───────────────────────────────────────────────────────────────────
print("\n>>> CICLO 1")
texto1 = "quero entender como a atenção funciona nesse sistema"
log1, ancora1 = processar(texto1, ancara=None)

# ── Ciclo 2 (com âncora do ciclo 1) ──────────────────────────────────────────
print("\n>>> CICLO 2  (âncora ativa — pegada do ciclo anterior)")
texto2 = "a frequência define quem responde ao professor"
log2, ancora2 = processar(texto2, ancara=ancora1)

# ── Comparação dos picos ──────────────────────────────────────────────────────
print("\nCOMPARAÇÃO DE CICLOS")
print(f"  Ciclo 1  →  f_peak={log1.f_peak:.4f}   tokens passaram={log1.tokens_passaram}")
print(f"  Ciclo 2  →  f_peak={log2.f_peak:.4f}   tokens passaram={log2.tokens_passaram}")
print(f"  Desvio de pico: {abs(log2.f_peak - log1.f_peak):.4f}")
print(f"  Âncora transferida com intensidade={ancora1.intensidade:.4f}")
