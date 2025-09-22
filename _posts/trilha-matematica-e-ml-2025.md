---
layout: post
title: "Minha Trilha de Matemática + Machine Learning"
date: 2025-09-22 06:00:00 -0300
categories: [Aprendizado, Ciência de Dados, Engenharia]
tags: matemática machine-learning deep-learning álgebra cálculo estatística probabilidade python estudo portfólio
image: "/assets/img/ml-math-roadmap.jpg"
---

### Objetivo

Construir **base matemática sólida** e **habilidade prática** em machine learning, entendendo o que acontece por baixo das bibliotecas. A meta é sair do consumo de tutoriais para **projetos reais**, com código limpo, métricas claras e aprendizado contínuo.

### Princípios

- **Constância mínima:** 6–8h por semana, agendadas.  
- **Entender antes de automatizar:** implementar do zero antes de usar uma biblioteca.  
- **Memória ativa:** flashcards e revisão espaçada.  
- **Projetos pequenos, frequentes:** cada mês termina com um artefato público (repo/post).  
- **Métricas claras:** loss, acurácia, F1, curva de aprendizado e _error analysis_.

### Roteiro — Meses

### Outubro 2025 — Álgebra Linear na Prática
- Vetores, matrizes, transformações lineares e decomposições (SVD, eigens).  
- Operações com `numpy` e visualização de bases/rotações.  
- Projeto: **redução de dimensionalidade** (PCA do zero) + comparação com `scikit-learn`.

### Novembro 2025 — Cálculo e Otimização
- Derivadas, gradientes, Jacobianas e regra da cadeia.  
- Otimizadores (GD, SGD, Momentum, Adam) e taxa de aprendizado.  
- Projeto: **regressão linear** do zero com _autograd_ manual e gráficos de superfície/contorno.

### Dezembro 2025 — Probabilidade & Estatística
- Variáveis aleatórias, distribuições, esperança, variância e intervalos de confiança.  
- Testes de hipótese, _bootstrap_ e análise exploratória de dados.  
- Projeto: **simulações** (Monte Carlo) e um relatório EDA com _notebook_ reproduzível.

### Janeiro 2026 — Fundamentos de ML Clássico
- _Train/validation/test_, validação cruzada, regularização (L1/L2), _bias-variance_.  
- Modelos: regressão logística, k-NN, SVM, árvores e _ensembles_.  
- Projeto: **pipeline clássico end-to-end** (dados reais) com _error analysis_.

### Fevereiro 2026 — Redes Neurais do Zero
- Perceptron, MLP, funções de ativação, inicialização e normalização.  
- _Backpropagation_ implementado manualmente; comparação com PyTorch.  
- Projeto: **classificador MLP** para imagens/tabulares, com _learning curves_.

### Março 2026 — Deep Learning Aplicado
- CNNs para visão, _sequence models_ (RNN/GRU/LSTM) e visão geral de _Transformers_.  
- _Fine-tuning_ em PyTorch (visão ou NLP) com _early stopping_ e _checkpointing_.  
- Projeto: **fine-tuning** em dataset público com relatório técnico.

### Abril 2026 — Representações & NLP Essencial
- Vetorização clássica (TF–IDF), _embeddings_ e _transfer learning_.  
- Pré-processamento de texto, avaliação (F1, ROC-AUC por classe) e _error analysis_.  
- Projeto: **classificação de texto** com comparativo TF–IDF vs. _embeddings_.

### Maio 2026 — Boas Práticas & MLOps Leve
- Versionamento de dados (DVC ou equivalente), experiment tracking e reprodução.  
- Servir modelo (API simples) e monitoramento inicial de inferência.  
- Projeto: **protótipo de serviço de ML** (API + README de deploy).

### Junho 2026 — Capstone
- Escolha de domínio (saúde, finanças, educação ou outro de interesse).  
- Coleta/limpeza, modelagem, métricas, explicabilidade e _trade-offs_.  
- Entrega: **repositório completo**, _post_ técnico e apresentação curta.

## Stack Mínima

- **Linguagem:** Python 3.11+  
- **Bibliotecas:** `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `pytorch` (a partir de fev/2026)  
- **Ambiente:** Jupyter + `venv`/`conda`, `pre-commit` (linters/formatters), `git` com _issues_ e _milestones_.  
- **Opcional:** GPU em _cloud_ para _fine-tuning_.

## Rotina Semanal (sugestão)

- 2× sessões de 90–120 min (teoria + exercícios).  
- 1× sessão de 120 min (projeto/código do zero).  
- 30 min de revisão espaçada (flashcards/notas) e 15 min de _retrospective_.

## Métricas de Progresso

- **Teoria:** flashcards respondidos/semana, exercícios resolvidos.  
- **Prática:** PRs abertas/fechadas, _issues_ concluídas e _notebooks_ reproduzíveis.  
- **Qualidade:** cobertura de testes dos utilitários, _lint_ limpo, README com _quickstart_.  
- **Impacto:** 1 projeto público/mês (repo + post).

## Sobre o Rumo

Não é sobre “usar a biblioteca certa”, mas **saber por que ela funciona**. O objetivo é ganhar clareza conceitual, disciplina de experimentação e um portfólio que demonstre raciocínio, não apenas resultados. Se eu mantiver a constância, a profundidade vem como consequência — e os projetos contarão essa história.
