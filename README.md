# analog-attention

**v0.1.0** · GPL-3.0  
**Autor / Arquiteto:** Zaqueu Ribeiro da Costa

---

## O que é isso?

`analog-attention` é um mecanismo de atenção baseado em **assinaturas de frequência**, não em produto de matrizes. É parte de uma arquitetura nova de IA — não compete com Transformers, constrói um conceito diferente.

A ideia central: cada palavra tem uma **assinatura de onda**, como uma impressão digital de frequência. A atenção acontece por ressonância entre frequências — não por comparar todos os pares de tokens.

---

## Por que isso é diferente de um Transformer?

| Transformer clássico | analog-attention |
|---|---|
| Compara todos os tokens entre si — O(n²) | Assinatura por frequência — O(n) |
| Similaridade por produto escalar | Ressonância parabólica de frequências |
| Embeddings aprendidos (gradiente) | Assinaturas calculadas (sem treino) |
| Precisa de GPU e milhões de parâmetros | Roda em tempo real, leve |
| Resposta via predição de tokens | Resposta via estado interno (NEC) |

---

## Como funciona — fluxo completo

```
Texto de entrada
      ↓
  Tokenizador
  (cada palavra vira um Token com assinatura de frequência)
      ↓
  BASE (Base de Assinaturas de Onda)
  (classifica cada token em uma classe de onda: ONDA_INFRA → ONDA_ULTRA_ALTA)
      ↓
  Broadcast Parabólico
  (tokens com frequência próxima ao pico da query recebem mais sinal)
      ↓
  Propagação Harmônica em Cadeia
  (tokens ativados propagam sinal para vizinhos harmonicamente similares)
      ↓
  Log de Frequência + Âncora de Contexto
  (resultado do ciclo + memória para o próximo ciclo)
      ↓
  NEC — Núcleo Emocional-Cognitivo
  (gera resposta por estado interno, sem LLM)
```

---

## Estrutura de arquivos

```
analog_attention/
├── signature.py      — Calcula assinatura de frequência (histograma de caracteres → FFT)
├── base.py           — BASE: mapa estático de classes de onda (8 classes, frequência 0→1)
├── token.py          — Token: unidade com assinatura, fase, classe de onda, sinal recebido
├── tokenizer.py      — Tokenizador: texto → lista de Tokens classificados
├── broadcast.py      — Broadcast parabólico: σ = I · max(0, 1 − κ · (f_token − f_peak)²)
├── candidate.py      — Filtra tokens com sinal acima do limiar
├── propagate.py      — Propagação em cadeia harmônica (cosine similarity entre assinaturas)
├── output.py         — FrequencyLog + AncoradeContexto (memória entre ciclos)
├── pipeline.py       — Pipeline completo: texto → log + âncora
├── scan.py           — Varredura inicial de tokens
└── compose.py        — Composição de resultado final

analog_attention/nec/
├── emocional.py      — Estágio 01: Estrutura Emocional (valência, intensidade, coerência...)
├── estado.py         — Estágio 02: Estado Atual + detector de salto de contexto
├── associativo.py    — Estágio 03: Referência Associativa (delta_novidade, D1/D2)
├── operador.py       — Operador Θ: combina EE + EA + RA → campo P normalizado
├── projecao.py       — Estágio 04: Projeção de Possíveis (seleciona modo de resposta)
├── geracao.py        — Estágio 05: Gera resposta por estado (sem LLM)
└── nucleo.py         — Orquestrador: roda todos os estágios em sequência

examples/
├── prototipo_entrada.py  — Exemplo básico de pipeline (entrada de texto)
├── nec_demo.py           — Demo do NEC com 3 ciclos e âncora de contexto
└── visualizador.html     — Visualizador interativo (abre no navegador, sem servidor)

tests/
├── test_signature.py     — Testes da assinatura de frequência
├── test_base.py          — Testes da BASE de classes de onda
├── test_tokenizer.py     — Testes do tokenizador
├── test_pipeline.py      — Testes do pipeline completo
└── test_nec.py           — Testes do NEC (10 testes)
```

---

## Conceitos principais explicados

### Assinatura de frequência

Cada token (palavra) é convertido em um vetor de frequência usando histograma de caracteres. A frequência dominante é o centroide espectral — um número entre 0 e 1 que representa "onde" essa palavra vive no espectro.

Palavras com caracteres de baixo código ASCII tendem a frequências baixas. Palavras com caracteres de alto código tendem a frequências altas. Isso é determinístico — não precisa de treino.

### BASE (Base de Assinaturas de Onda)

8 classes de onda cobrindo o espectro [0, 1]:

| Classe | Faixa | Significado |
|---|---|---|
| ONDA_INFRA | 0.000 – 0.100 | Frequência mínima |
| ONDA_ULTRA_BAIXA | 0.100 – 0.200 | |
| ONDA_BAIXA | 0.200 – 0.350 | |
| ONDA_MEDIA_BAIXA | 0.350 – 0.500 | |
| ONDA_MEDIA | 0.500 – 0.650 | |
| ONDA_MEDIA_ALTA | 0.650 – 0.800 | |
| ONDA_ALTA | 0.800 – 0.920 | |
| ONDA_ULTRA_ALTA | 0.920 – 1.000 | Frequência máxima |

### Broadcast parabólico

O "professor" (query) emite um sinal. Tokens com frequência próxima ao pico da query recebem mais sinal. A curva é parabólica — tokens distantes no espectro recebem zero.

```
σ = I · max(0, 1 − κ · (f_token − f_peak)²)
```

- `I` = intensidade
- `κ` = curvatura (quão seletiva é a convocação)
- `f_peak` = frequência dominante da query

### Âncora de contexto (dois ciclos)

Ao final de cada ciclo, o sistema gera uma âncora — uma memória da assinatura agregada dos tokens ativados. No ciclo seguinte, a query é misturada com essa âncora:

```
query_nova = (1 − α) · query + α · âncora
```

Isso cria continuidade entre turnos, como um fio que conecta os ciclos sem reiniciar o estado.

### NEC — Núcleo Emocional-Cognitivo

O NEC é um bloco de 5 estágios que processa a entrada e gera uma resposta **por estado interno**, sem LLM:

1. **Estrutura Emocional** — calcula valência, intensidade, coerência, ativação, carga a partir das frequências dos tokens ativados
2. **Estado Atual** — detecta se o contexto mudou de natureza (`flag_salto`) ou continua em fluxo
3. **Referência Associativa** — mede novidade (`delta_novidade`) e separa tokens em D1 (familiar novo) e D2 (familiar conhecido)
4. **Projeção de Possíveis** — seleciona modo: `exploratório`, `confirmatório`, `adaptativo`, `tenso` ou `estável`
5. **Geração por Estado** — monta resposta a partir de templates do modo + tokens-chave ativados

O Operador Θ combina os três primeiros estágios em um campo P normalizado que guia a geração.

---

## Como rodar

```bash
# Instalar
pip install -e .

# Pipeline básico
python examples/prototipo_entrada.py

# Demo do NEC com múltiplos ciclos
python examples/nec_demo.py

# Testes
python -m pytest tests/ -v

# Visualizador interativo
# Abrir examples/visualizador.html no navegador
```

---

## Estado atual

| Módulo | Status |
|---|---|
| Tokenizador + BASE | estável |
| Assinatura de frequência | estável |
| Broadcast parabólico | estável |
| Propagação harmônica | estável |
| Âncora de contexto | estável |
| Pipeline completo | estável |
| NEC v0.1 | funcionando |
| Visualizador HTML | funcionando |

**37 testes passando.**

---

## Roadmap

- **v0.1** — base completa, NEC funcional (atual)
- **v0.2** — refinamento da geração por estado, schema BASA real
- **v0.3** — interface de tempo real, benchmarks de frequência
- **v1.0** — publicação, documentação técnica completa

---

## Licença

GPL-3.0 — Qualquer uso público, acadêmico ou comercial deve incluir crédito visível ao autor:

**Zaqueu Ribeiro da Costa**
