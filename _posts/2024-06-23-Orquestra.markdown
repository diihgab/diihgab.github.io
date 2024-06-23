---
layout: post
title: "Orquestração de Fluxo de Trabalho com Apache Doris Job Scheduler"
date: 2024-06-23 11:00:00 -0300
categories: [BigData, Automatização]
tags: apachedoris jobscheduler automatizaçãodedados gestãoeficiente bigdata dataautomation datamanagement
image: "/assets/img/orquestra.PNG"
---

No artigo de hoje irei abordar um tema crucial para o *gerenciamento de dados*: **o agendamento de tarefas**. Vamos falar sobre uma ferramenta que promete transformar sua rotina de gerenciamento de dados em algo mais automatizado e eficiente – o [Doris Job Scheduler](https://doris.apache.org/) . Se você está cansado de ficar lembrando de cada atualização ou limpeza de dados, prepare-se para conhecer uma solução que vai fazer tudo isso por você. Vamos lá!

## **Por que usar o Doris Job Scheduler?**

O agendamento de tarefas é vital para manter seus dados atualizados e organizados. Ferramentas como [Apache Airflow](https://airflow.apache.org/?ref=hackernoon.com) e [Apache Dolphinscheduler](https://dolphinscheduler.apache.org/en-us?ref=hackernoon.com) são populares, mas convenhamos, adicionar mais uma ferramenta à sua arquitetura significa mais trabalho de gerenciamento e manutenção. É aqui que o **Doris Job Scheduler voa mlk**. Integrado ao [Apache Doris](https://doris.apache.org/blog/release-note-2.1.0/?ref=hackernoon.com), ele oferece *flexibilidade* e *simplicidade*, eliminando a necessidade de componentes extras. Afinal, quem não quer menos complicação na vida?

## **Principais Recursos**

Vamos destacar alguns dos recursos que fazem do **Doris Job Scheduler** uma ferramenta indispensável:

- **Eficiência:** Utiliza o algoritmo TimeWheel para garantir que suas tarefas sejam disparadas com precisão de até um segundo. Porque, obviamente, cada segundo conta.
- **Flexibilidade:** Suporta trabalhos únicos e recorrentes, com horários de início e término customizáveis. Porque todos amamos opções, certo?
- **Execução eficiente:** Baseado em um modelo de produtor único e multiconsumidor, evitando sobrecarga na execução das tarefas. Menos travamento, mais produtividade.
- **Rastreabilidade:** Mantém registros das execuções de tarefas que podem ser consultados facilmente. Transparência é tudo.
- **Alta disponibilidade:** Altamente disponível e recuperável. Porque falhar não é uma opção... até que seja.
Como Funciona?
Criar e agendar tarefas no *Doris* é *surpreendentemente simples*. Vamos explorar a *estrutura básica* e ver alguns exemplos práticos para você entender como isso funciona na prática.

# **Criando e Agendando Tarefas**
Aqui está a estrutura para criar um trabalho:

```sql
CREATE TRABALHO nome_trabalho
ON SCHEDULE agendamento
[COMMENT 'comentário']
DO execute_sql;
```

# **Elementos da Instrução**

- **CREATE TRABALHO:** Define o nome do trabalho.
- **ON SCHEDULE:** Especifica o tipo e a frequência do trabalho.
- **AT timestamp:** Para agendamentos únicos.
- **EVERY:** Para trabalhos recorrentes.
- **DO:** Define a ação a ser executada, atualmente suportando **INSERT**.

# **Exemplos Práticos**

Vamos tornar isso mais concreto com alguns exemplos:

**Exemplo 1: Trabalho recorrente a cada minuto**

```sql
CREATE TRABALHO meu_trabalho ON SCHEDULE EVERY 1 MINUTE DO INSERT INTO bd1.tbl1 SELECT * FROM bd2.tbl2;
```

**Exemplo 2: Trabalho único em uma data específica**

```sql
CREATE TRABALHO meu_trabalho ON SCHEDULE AT '2025-01-01 00:00:00' DO INSERT INTO bd1.tbl1 SELECT * FROM bd2.tbl2;
```

**Exemplo 3: Trabalho recorrente diário a partir de uma data**

```sql
CREATE TRABALHO meu_trabalho ON SCHEDULE EVERY 1 DAY STARTS '2025-01-01 00:00:00' DO INSERT INTO bd1.tbl1 SELECT * FROM bd2.tbl2 WHERE create_time >= days_add(now(), -1);
```

**Sincronização Automática de Dados**

Agora, vamos falar sobre algo realmente *emocionante*: **sincronização automática de dados**. Combinando o *Job Scheduler* com o recurso [Multi-Catalog](https://doris.apache.org/docs/lakehouse/lakehouse-overview/?ref=hackernoon.com#multi-catalog) do *Apache Doris*, você pode sincronizar dados de diferentes fontes sem precisar se descabelar. Isso é especialmente útil para usuários de *e-commerce* que precisam importar regularmente dados do *MySQL* para análise no *Doris*.

# **Passo a Passo**
**Passo 1: Criar uma tabela no *Doris***

```sql
CREATE TABLE IF NOT EXISTS atividade_usuario (
    `id_usuario` LARGEINT NOT NULL COMMENT "ID do Usuário",
    `data` DATE NOT NULL COMMENT "Data da importação de dados",
    `cidade` VARCHAR(20) COMMENT "Cidade do usuário",
    `idade` SMALLINT COMMENT "Idade do usuário",
    `sexo` TINYINT COMMENT "Sexo do usuário",
    `data_ultima_visita` DATETIME REPLACE DEFAULT "1970-01-01 00:00:00" COMMENT "Data da última visita do usuário",
    `custo` BIGINT SUM DEFAULT "0" COMMENT "Quantia total de consumo do usuário",
    `tempo_maximo` INT MAX DEFAULT "0" COMMENT "Tempo máximo de permanência do usuário",
    `tempo_minimo` INT MIN DEFAULT "99999" COMMENT "Tempo mínimo de permanência do usuário"
)
AGGREGATE KEY(`id_usuario`, `data`, `cidade`, `idade`, `sexo`)
DISTRIBUTED BY HASH(`id_usuario`) BUCKETS 1
PROPERTIES ("replication_allocation" = "tag.location.default: 1");
```

**Passo 2: Criar um catálogo no *Doris* para o *MySQL***

```sql
CREATE CATALOG atividade PROPERTIES (
    "type"="jdbc",
    "user"="root",
    "jdbc_url" = "jdbc:mysql://127.0.0.1:9734/usuario?useSSL=false",
    "driver_url" = "mysql-connector-java-5.1.49.jar",
    "driver_class" = "com.mysql.jdbc.Driver"
);
```

**Passo 3: Ingerir dados do *MySQL* para *Doris***

**Trabalho único para carga massiva de dados**

```sql
CREATE TRABALHO trabalho_unico
ON SCHEDULE AT '2024-08-10 03:00:00'
DO INSERT INTO atividade_usuario FROM SELECT * FROM atividade.usuario.atividade;
```

**Trabalho regular para atualização diária**

```sql
CREATE TRABALHO carga_agendada
ON SCHEDULE EVERY 1 DAY
DO INSERT INTO atividade_usuario FROM SELECT * FROM atividade.usuario.atividade WHERE create_time >= days_add(now(), -1);
```

# **Projeto Técnico e Implementação**

Para os entusiastas da engenharia, o **Doris Job Scheduler** combina o *algoritmo TimingWheel* com a estrutura *Disruptor*, garantindo agendamentos precisos e baixo uso de memória. Isso significa que você não vai se afogar em tarefas acumuladas e cada uma será executada no momento certo. Porque quem gosta de travamentos e atrasos, não é?

# **Aplicações do Doris Job Scheduler**

O **Doris Job Scheduler** é um verdadeiro canivete suíço. Além de ser útil para *ETL* e *análise de data lake*, ele é fundamental para a implementação de visualizações materializadas assíncronas[^assincronas], garantindo que as atualizações sejam realizadas *periodicamente* e de forma *consistente*. Um *verdadeiro sonho*, não acha?

# **O Futuro do Doris Job Scheduler**

A **comunidade Apache Doris** está constantemente inovando. E o que vem por aí? Prepare-se para:

- **Exibição da distribuição das tarefas na WebUI.**
- **Trabalhos em formato DAG para orquestração de tarefas complexas.**
- **Suporte para mais operações como UPDATE e DELETE.**

Então, o que acha? Pronto para adotar o **Doris Job Scheduler** e automatizar suas tarefas de uma maneira eficiente e, por que não, um tanto quanto divertida? Suas tarefas estão esperando para serem agendadas! :)

Obrigado por ler até aqui! **- Diego Gabs**

---

[^assincronas]: Sobre Códigos Assíncronos: https://diihgab.github.io/posts/DesignPatternsNodeJS/