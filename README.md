analog-attention

v0.1.0 · GPL-3.0

Propagação de Atenção por Coordenadas de Token

---

👤 Autor / Arquiteto

Zaqueu Ribeiro da Costa

---

📌 Conceito Central

Em vez de calcular similaridade entre todos os pares de tokens (O(n²)), o sistema realiza uma varredura inicial que constrói um mapa de coordenadas. A partir desse mapa, a atenção é direcionada por endereçamento direto, não por busca.

---

🧠 1. Visão Geral

"analog-attention" é uma biblioteca Python que implementa um mecanismo de atenção baseado em propagação por coordenadas, inspirado na dinâmica social de uma sala de aula.

1.1 Problema da Atenção Clássica

Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V

- Complexidade: O(n²)
- Cada token compara com todos os outros → alto custo e redundância

---

1.2 Proposta

Substituição por um protocolo em duas fases principais:

- Scan (varredura) → O(n)
  Constrói mapa de coordenadas (feito uma única vez)

- Broadcast (endereçamento direto) → O(k)
  Convoca tokens específicos

- Propagate (propagação social) → O(k · r)
  Tokens repassam sinal localmente

---

🎓 Analogia

«Professor chama alunos → alunos próximos propagam → surge um padrão coletivo»

---

🏗️ 2. Arquitetura

2.1 Componentes

Componente| Função
Token| Unidade com coordenada e sinal
scan()| Cria mapa de coordenadas
broadcast()| Envia sinal inicial
candidate()| Filtra tokens relevantes
propagate()| Propagação social
compose()| Agrega resultado

---

2.2 Fluxo

INPUT
  ↓
scan(tokens)         → O(n)
  ↓
broadcast()          → O(k)
  ↓
candidate()          → O(k)
  ↓
propagate()          → O(k·r)
  ↓
compose()
  ↓
OUTPUT

---

2.3 Complexidade

Sistema| Complexidade
Transformer| O(n²)
Sparse Attention| O(n log n)
analog-attention| O(n) + O(k) + O(k·r)

---

🔌 3. API Pública

Classe Token

class Token:
    def __init__(self, nome: str, coordenada: tuple[float, float])
    def receber_sinal(self, sinal: float) -> None

    sinal_recebido: float
    foi_atingido: bool

---

scan()

def scan(tokens: list[Token]) -> dict[str, tuple]:
    """Retorna mapa de coordenadas"""

---

broadcast()

def broadcast(intensidade, token, ruido=0.0, falloff='sqrt'):

---

candidate()

def candidate(token, threshold=0.4) -> bool:

---

propagate()

def propagate(tokens, intensidade_professor=5.0, ...):

---

⚙️ 4. Implementação Atual (v0.1.0)

import math

class Token:
    def __init__(self, nome, coordenada):
        self.nome = nome
        self.coordenada = coordenada
        self.sinal_recebido = 0

    def receber_sinal(self, sinal):
        self.sinal_recebido += sinal


def atenção_professor(intensidade, token, ruído=0):
    x, y = token.coordenada
    distancia = math.hypot(x, y)
    sinal = intensidade / (distancia**2 + 1) - ruído
    return max(sinal, 0)


def atenção_social(sinal_aluno, origem, vizinho, fator=0.5):
    x1,y1 = origem.coordenada
    x2,y2 = vizinho.coordenada
    distancia = math.hypot(x2-x1, y2-y1)
    sinal = sinal_aluno * fator / (distancia + 1)
    return max(sinal, 0)


def propagar_sinal(tokens, intensidade=5, ruído=0, fator_social=0.5):
    for token in tokens:
        sinal = atenção_professor(intensidade, token, ruído)
        token.receber_sinal(sinal)

    for token in tokens:
        if token.sinal_recebido > 0:
            for vizinho in tokens:
                if vizinho != token:
                    s = atenção_social(token.sinal_recebido, token, vizinho, fator_social)
                    vizinho.receber_sinal(s)

    return tokens

---

🔧 Pendências (v0.2.0)

1. Implementar "scan()" explícito
2. Adicionar raio de vizinhança
3. Implementar "candidate()" com threshold
4. Tornar falloff configurável

---

🚀 5. Roadmap

v0.1.0

- Base funcional
- Simulação inicial

v0.2.0

- Primitivas completas
- Otimização de complexidade

v0.3.0

- Integração com PyTorch / JAX
- Benchmarks

v1.0.0

- Publicação no PyPI
- Paper técnico

---

📐 6. Especificação Técnica

Scan

coordinate_map = { token.nome: token.coordenada }

---

Broadcast (falloff)

- Quadrático
- Raiz
- Linear

---

Candidate

σ ≥ threshold

---

Propagação

σᵢⱼ = σᵢ · φ / (d + 1)

---

Diferença para Transformer

Transformer| analog-attention
Busca global| Endereçamento direto
Similaridade| Coordenadas
O(n²)| O(k)
Softmax| Falloff espacial

---

📁 6.3 Estrutura de Pastas

analog-attention/
├── analog_attention/
├── extensions/
├── tests/
├── examples/
├── README.md
├── LICENSE
└── pyproject.toml

---

📜 7. Licença

GPL-3.0

Atribuição obrigatória:

Este software foi desenvolvido por:
Zaqueu Ribeiro da Costa

Qualquer uso público, acadêmico ou comercial deve incluir
crédito visível ao autor.

Você pode:

- Usar
- Modificar
- Distribuir

Desde que mantenha a mesma licença.

---

🔄 Status

🚧 Em desenvolvimento ativo

---
