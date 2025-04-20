# **ForecastFC - Modelo de Machine Learning para Previsão de Resultados Futebolísticos**

![Imagem de capa](https://github.com/UFRJ-Analytica/PS-2025.1/blob/main/capa.png)

## Descrição
O objetivo deste desafio é que vocês realizem uma análise exploratória dos dados, buscando entender as relações entre as variáveis, fazer o tratamento dos dados e, por fim, desenvolver um modelo de IA para prever resultados de partidas de futebol. ⚽🤖

A base de dados está disponível nesse mesmo repositório, no diretório `Data/`. Seu grupo deve criar uma branch nesse repositório e desenvolver os códigos nela. (Também pode fazer um fork, se preferir). A entrega deve ser feita até o dia 20/04/2025.

O case é realmente bastante desafiador, então para nossos candidatos iniciantes: Não tenham medo, na hora de avaliá-los, não iremos comparar exclusivamente o aspecto técnico como única métrica. Queremos que este desafio seja primariamente uma oportunidade para que vocês possam se desenvolver, aprender e também tirar dúvidas com os demais candidatos para que todos possam evoluir juntos. O nível de experiência e conhecimento prévio de cada um será levado em consideração.

Além disso, também incluímos alguns materiais de apoio abaixo para ajudar nos aspectos mais técnicos, e também os encorajamos a usar as ferramentas que têm à disposição: Livros, Google, StackOverflow, LLMs como ChatGPT, etc.

Vocês foram divididos em esquadrões pra trabalharem em equipe, então não esqueçam de entrar em contato com o pessoal do seu grupo! 

Cada grupo deve clonar esse repositório e desenvolver o código de forma colaborativa, seguindo as boas práticas do Git (branches, commits bem descritos, pull requests, etc).

- [Material de apoio: Git](./Materiais/git.md)
- [Material de apoio: EDA (Análise explorartória dos dados)](./Materiais/EDA.md)
- [Material de apoio: Modelos de aprendizado de máquina](./Materiais/modelos-ApMaq.md)

Observe que NÃO é estritamente necessário implementar um modelo completamente funcional ao final da tarefa! Devido à variedade dos níveis de experiência dos candidatos, a tarefa abre margem para implementações bastante avançadas, mas a avaliação vai levar em consideração o nível técnico e experiência de cada um, não havendo critérios técnicos fixos.

Sendo assim, em muitos casos uma análise exploratória bem pensada pode ser uma entrega

## Descrição das coluna
Essa tabela dá o significado de cada coluna. Considere que “1” se refere ao time da casa e “2” ao time visitante:

<table>
  <tr>
    <th>Chutes a gol 1 / 2</th>
    <td>Número de finalizações que foram enquadradas (ao menos foram na direção do gol) pelo time 1 / time 2.</td>
  </tr>
  <tr>
    <th>Impedimentos 1 / 2</th>
    <td>Quantas vezes cada time foi pego em posição de impedimento.</td>
  </tr>
  <tr>
    <th>Escanteios 1 / 2</th>
    <td>Total de cobranças de escanteio a favor de cada equipe.</td>
  </tr>
  <tr>
    <th>Chutes fora 1 / 2</th>
    <td>Finalizações que não foram na direção do gol (para fora) de cada time.</td>
  </tr>
  <tr>
    <th>Faltas 1 / 2</th>
    <td>Quantas faltas cada time cometeu durante a partida.</td>
  </tr>
  <tr>
    <th>Cartões amarelos 1 / 2</th>
    <td>Quantos cartões amarelos foram mostrados a jogadores de cada time.</td>
  </tr>
  <tr>
    <th>Cartões vermelhos 1 / 2</th>
    <td>Quantos cartões vermelhos foram mostrados a jogadores de cada time.</td>
  </tr>
    <tr>
    <th>Cruzamentos 1 / 2</th>
    <td>Número de passes laterais elevados (cruzamentos) realizados por cada equipe.</td>
  </tr>
    <tr>
    <th>Laterais 1 / 2</th>
    <td>Quantas vezes cada time executou arremessos laterais.</td>
  </tr>
    <tr>
    <th>Chutes bloqueados 1 / 2</th>
    <td>Finalizações de cada time que foram bloqueadas por defensores adversários.</td>
  </tr>
  <tr>
    <th>Contra-ataques 1 / 2</th>
    <td>Quantas ações de contra-ataque (recuperação e transição rápida) cada equipe conduziu.</td>
  </tr>
  <tr>
    <th>Gols 1 / 2</th>
    <td>Número de gols marcados por cada time.</td>
  </tr>
  <tr>
    <th>Tiro de meta 1 / 2</th>
    <td>Quantos arremessos de meta (goal kicks) cada time cobrou.</td>
  </tr>
  <tr>
    <th>Tratamentos 1 / 2</th>
    <td>Quantas vezes jogadores de cada time receberam atendimento médico em campo.</td>
  </tr>
  <tr>
    <th>Substituições 1 / 2</th>
    <td>Número de trocas de jogadores realizadas por cada equipe.</td>
  </tr>
  <tr>
    <th>Tiros-livres 1 / 2</th>
    <td>Quantas cobranças de falta (tiros livres) cada time teve.</td>
  </tr>
  <tr>
    <th>Defesas difíceis 1 / 2</th>
    <td>Número de defesas de alta dificuldade feitas pelos goleiros de cada time.</td>
  </tr>
  <tr>
    <th>Posse 1 / 2 (%)</th>
    <td>Percentual de tempo de posse de bola de cada equipe ao longo da partida.</td>
  </tr>
  <tr>
    <th>Time 1 / 2</th>
    <td>Nome do time da casa (1) e do time visitante (2).</td>
  </tr>
  <tr>
    <th>Position 1 / 2</th>
    <td>Posição tática inicial ou formação de cada equipe (por exemplo: 4-4-2, 3-5-2 etc.).</td>
  </tr>
</table>
