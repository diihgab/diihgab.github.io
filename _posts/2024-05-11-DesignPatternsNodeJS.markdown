---
layout: post
title: "Design Patterns como Programação Assíncrona em NodeJS"
date: 2024-05-11 03:10:00 -0300
categories: [Arquitetura de Software, Sistemas Distribuídos]
tags: desenvolvimentoweb padroesassincronos callbackhell asyncawait designpatterns
image: "/assets/img/nodejs.jpeg"
---

Antes de começar, vamos a uma breve introdução ao **Design Patterns**, onde representam soluções estabelecidas e generalizadas para problemas comumente recorrentes encontrados durante o desenvolvimento de software, funcionando mais como *projetos conceituais* ou *modelos adaptáveis* ​​a vários contextos.

O *Design Patterns* oferece vários benefícios como:

- **Soluções reutilizáveis**: Abordagens comprovadas para problemas comuns, economizando tempo e esforço.
- **Melhor Qualidade de Código**: Maior clareza, capacidade de manutenção e flexibilidade.
- **Melhor Comunicação**: Vocabulário e compreensão compartilhados entre os desenvolvedores.

No NodeJS, onde tudo é executado de forma *assíncrona*, dominar as *patterns assíncronas* é crucial, pontuarei os principais motivos:

**1. Callbacks:** Os clássicos callbacks, aqui as funções são passadas como argumentos para operações assíncronas que são chamadas após a finalização. Onde pode ocasionar a famosa *"callback hell"* (calma que irei explicar com mais detalhes quando chegarmos lá) onde as funções são aninhadas, mas ainda é amplamente utilizado.

**2. Promises:** É oferecido uma maneira mais limpa de gerenciar o fluxo assíncrono. Representam o resultado eventual *(sucesso ou falha)* de uma operação, permitindo encadeamento e tratamento de erros.

**3. Async/Await:** Um patamar acima do *Promises*, onde o código assíncrono fica bem parecido com um código síncrono. Onde as funções `async` e `await` podem garantir um fluxo mais limpo.

Agora que demos uma breve introdução sobre as três *Patterns* mais utilizadas, podemos começar o artigo :)
Onde veremos com mais detalhes essas três *patterns assíncronas* atuando no *NodeJS* com mais detalhes.

# **Callback Pattern**

O *Callback Pattern* (também conhecido como *Retorno de Chamada*) é um conceito fundamental em *NodeJS*, permitindo lidar com operações assíncronas com eficiência. Envolve passar uma função *(o Callback)* como argumento para outra função, que então invoca o nosso *Callback* após a conclusão de sua tarefa assíncrona. Isso permite que seu código continue executando outras operações enquanto a tarefa assíncrona está em andamento, promovendo comportamento sem bloqueio e uso eficiente do loop de eventos.

## **Principais casos de uso**

- **Operações de E/S de arquivo:** Leitura ou gravação em arquivos de forma assíncrona.
- **Solicitações de rede:** Busca de dados de APIs ou servidores.
- **Manipulação de eventos:** Respondendo a interações do usuário ou eventos do sistema.
- **Interações de banco de dados:** Recuperação ou manipulação de dados de bancos de dados.
- **Agendamento de tarefas:** Execução de funções após um atraso ou em intervalos específicos.

## **Exemplos de código**
Um uso básico de *Callback*:

```javascript
fs.readFile('arquivo.txt', (err, data) => {
  if (err) {
    console.error('erro ao ler arquivo:', err);
  } else {
    console.log('conteudo do arquivo:', data.toString());
  }
});
```

Encadeando *Callbacks*:

```javascript
function getUser(userId, callback) {
  // Simula a busca de dados do usuario de forma assincrona
  setTimeout(() => {
    callback(null, { id: userId, name: 'Diego Gabriel' });
  }, 1000);
}

function getPostsForUser(userId, callback) {
  // Simula a busca de postagens de usuários de forma assíncrona
  setTimeout(() => {
    callback(null, [{ title: 'Post 1' }, { title: 'Post 2' }]);
  }, 500);
}

getUser(1, (err, user) => {
  if (err) {
    console.error('Erro ao obter usuário:', err);
  } else {
    getPostsForUser(user.id, (err, posts) => {
      if (err) {
        console.error('Erro ao obter postagens:', err);
      } else {
        console.log('Usuário:', user);
        console.log('Posts:', posts);
      }
    });
  }
});
```

**Prós**
- **Simplicidade:** Fácil de entender e implementar para operações assíncronas básicas.
- **Flexível:** Pode ser usado para diversos cenários assíncronos.
- **Conceito familiar:** Presente em muitas linguagens de programação e frameworks.

**Contras**
- **Callback Hell:** Esse tipos de retornos de chamada aninhados podem levar a códigos complexos e difíceis de ler.
- **O tratamento de erros pode ser complicado:** Gerenciar erros em *Callbacks* aninhados pode ser desafiador.
- **Difícil de testar:** Testar código com *Callbacks* pode ser mais complexo.

# **Promise Pattern**

O *Promise Pattern* fornece um mecanismo para lidar com operações assíncronas de uma forma mais estruturada e legível em comparação com os *Callbacks* vistos anteriormente. Representa a eventual conclusão (ou falha) de uma operação assíncrona e permite encadear ações com base em seu resultado. Isso promove uma organização de código mais limpa e facilita o tratamento de erros, melhorando o fluxo e a capacidade de manutenção do código assíncrono.

## **Principais casos de uso**

- **Encadeamento de operações assíncronas:** Execução sequencial de várias tarefas assíncronas, uma após a outra.
- **Tratamento de resultados assíncronos:** Acessar e reagir aos resultados de operações assíncronas quando elas forem concluídas.
- **Tratamento de erros:** Tratamento elegante de erros que podem ocorrer durante operações assíncronas.
- **Melhorando a legibilidade do código:** Estruturando código assíncrono de maneira mais declarativa e previsível.
- **Melhorando a testabilidade:** Simplificando o teste de código assíncrono isolando operações baseadas em *Promise*.

## **Exemplos de código**
Uso básico do *Promise*:

```javascript 
const fs = require('fs');

function readFileAsync(nome do arquivo) {
  return new Promise((resolver, rejeitar) => {
    fs.readFile(filename, 'utf8', (errar, dados) => {
      if (errar) {
        reject(errar);
      } else {
        resolve(dados);
      }
    });
  });
}
readFileAsync('arquivo.txt')
  .then((data) => {
    console.log('Conteúdo do arquivo:', data);
  })
  .catch((err) => {
    console.error('Erro ao ler arquivo:', err);
  });

```

Encadeando *Promises*:

```javascript
function getUser(userId) {
  return new Promise((resolver, rejeitar) => {
    // Simula a busca de dados do usuário
    setTimeout(() => {
      resolve({ id: userId, name: 'Diego Gabriel' });
    }, 1000);
  });
}

function getPostsForUser(userId) {
  return new Promise((resolver, rejeitar) => {
    // Simula a busca de postagens do usuário
    setTimeout(() => {
      resolve([{ title: 'Post 1' }, { title: 'Post 2' }]);
    }, 500);
  });
}

getUser(1)
  .then((usuario) => {
    console.log('Usuário:', usuario);
    return getPostsForUser(usuario.id);
  })
  .then((posts) => {
    console.log('Postagens do usuário:', posts);
  })
  .catch((err) => {
    console.error('Erro:', err);
  });

  ```

**Pros**

- **Legibilidade aprimorada:** Estrutura de código mais clara em comparação com *callbacks* aninhados.
- **Tratamento de erros simplificado:** Mecanismo centralizado com relação aos tratamentos de erros.
- **Capacidades de encadeamento:** Permite fácil sequenciamento de operações assíncronas.
- **Testabilidade aprimorada:** Mais fácil de isolar e testar unidades assíncronas.
- **Sintaxe async/await:** Fornece uma sensação mais síncrona para código assíncrono.

**Contras**

- **Curva de aprendizado potencial:** Compreender as *Promises* inicialmente requer esforço. (Eu pelo menos demorei kkkk)
- **Evita Callback Hell:** Não é uma solução mágica, ainda é necessário um planejamento cuidadoso.
- **Risco de uso excessivo:** Nem toda operação assíncrona requer *Promises*.

# **Async/Await Pattern**

O *Async/Await Pattern* baseia-se em *Promises*, oferecendo uma sintaxe semelhante ao código síncrono para lidar com operações assíncronas. Usando essas palavrinhas-chaves: `async` e `await`, você pode escrever código que parece ser executado sequencialmente, mesmo que envolva etapas assíncronas. Isso promove um código mais limpo e legível em comparação com *Promises* ou *Callbacks*, melhorando a capacidade de manutenção e a experiência do dev.

## **Principais casos de uso:**

- **Simplificando o código assíncrono:** Escrevendo código assíncrono que se parece mais com código síncrono.
- **Melhorando a legibilidade do código:** Mais fácil de entender o fluxo de operações assíncronas.
- **Reduzindo o Callback Hell:** Evitando *Callbacks* aninhados, resultando em um código mais limpo e menos sujeito a erros.
- **Tratamento de erros:** Aproveitando mecanismos existentes de tratamento de erros baseados em *Promise*.
- **Encadeamento de operações assíncronas:** Sequenciar tarefas assíncronas de maneira clara e concisa.

## **Amostras de código**
Uso básico do *Async/Await*:

```javascript
async function getUser(userId) {
  const resposta = await fetch(`https://api.example.com/users/${userId}`);
  const data = await resposta.json();
  return data;
}

(async () => {
  try {
    const user = await getUser(1);
    console.log('Usuário:', user);
  } catch (error) {
    console.error('Error:', error);
  }
})();

```

**Encadeando operações Async/Await**:

```javascript
async function fetchAndProcessData(url) {
  const response = await fetch(url);
  const data = await response.json();
  const processedData = processData(data); // Simula o processamento de dados
  return processedData;
}

(async () => {
  const data1 = await fetchAndProcessData('https://api.example.com/data1');
  const data2 = await fetchAndProcessData('https://api.example.com/data2');
  const combinedData = combineData(data1, data2); // Simula combinação de dados
  console.log('Combinação de dados:', combinedData);
})();
```

**Prós**

- **Legibilidade aprimorada:** A sintaxe se assemelha ao código síncrono, melhorando a clareza do código.
- **Complexidade reduzida:** Evita *Callbacks* aninhados, simplificando o fluxo de código.
- **Aproveita Promises:** Beneficia-se dos mecanismos existentes de tratamento de erros e encadeamento do *Promise*.
- **Experiência de desenvolvimento aprimorada:** torna o trabalho com código assíncrono mais intuitivo.
- **Melhor capacidade de manutenção:** O código fica mais fácil de entender e modificar.

**Contras**

- **Limitado as funções async:** Somente funções declaradas com `async` podem usar `await`.
- **Nuances de tratamento de erros:** `try...catch` ainda são necessários para tratamento de erros nas funções `async`.
- **Possível uso indevido:** o uso excessivo de `await` pode bloquear o loop de eventos, impactando o desempenho.

Isso é tudo sobre *Design Patterns* e suas relações de assincronicidade em *NodeJS*.

Obrigado por ler até aqui! **- Diego Gabs**