"""Exemplo: sala de aula com 6 alunos.

Professor chama a turma com intensidade 5.0.
Tokens perto da origem (professor) recebem mais sinal
e propagam para vizinhos dentro do raio.
"""
from analog_attention import Token, propagate, compose

tokens = [
    Token("Ana",    ( 1.0,  1.0)),
    Token("Bruno",  (-1.0,  1.0)),
    Token("Carla",  ( 0.0,  2.5)),
    Token("Diego",  ( 2.0,  4.0)),
    Token("Elena",  (-2.0,  4.0)),
    Token("Felipe", ( 0.0,  7.0)),
]

propagate(
    tokens,
    intensidade_professor=5.0,
    falloff="sqrt",
    fator_social=0.5,
    threshold=0.4,
    raio=3.0,
)

resultado = compose(tokens)

print(f"Sinal total do cluster: {resultado['sinal_total']:.4f}\n")
print(f"{'Token':<10} {'Sinal':>8}  {'Peso':>6}")
print("-" * 30)
for token in sorted(tokens, key=lambda t: t.sinal_recebido, reverse=True):
    peso = resultado["pesos"].get(token.nome, 0.0)
    atingido = "✓" if token.foi_atingido else " "
    print(f"{token.nome:<10} {token.sinal_recebido:>8.4f}  {peso:>6.2%}  {atingido}")
