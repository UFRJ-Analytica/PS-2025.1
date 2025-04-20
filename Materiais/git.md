# Git

## Introdução

O Git é um dos sistemas de controle de versão mais utilizados no mundo de desenvolvimento de software, ele possibilita que você veja e modifique o código de outros colaboradores, com mecanismos para impedir que o trabalho de um de vocês interfira no do outro. 

As adições ou remoções de código feitas por outros colaboradores são mostradas, de forma que você acompanhe o que tem sido feito no código. Além disso, é mantido um histórico de todas as alterações, o que pode ser útil no caso de uma atualização ter um bug, por exemplo, permitindo que você resgate uma versão anterior que funcionava. 

O git é tipicamente usado por linha de comando (CLI), e a instalação varia conforme o sistema operacional. Para windows, é possível instalá-lo nesse [link](https://git-scm.com/downloads/win). 

Para linux, os gerenciadores de pacote mais populares (como apt, pacman ou dnf) costumam incluí-lo. Dessa forma, você só precisa executar

```shell
sudo apt install git
```

ou o equivalente para sua distribuição. Se tiver dificuldades com a instalação (ou em usar a linha de comando), peça ajuda a um dos coordenadores! (Leão ou Jefferson)

## Versionamento
Um _repositório_ é um diretório (ou pasta) gerenciado pelo git. A partir do momento que um repositório git é iniciado, você pode salvar o estado dele fazendo um _commit_. Um commit é como salvar um ponto no tempo: Consiste em guardar todo o conteúdo do diretório (incluindo arquivos e subdiretórios) no histórico do git. Dessa forma, você pode recuperar esse estado a qualquer momento voltando para esse commit.

### Criando um repositório
Para iniciar um repositório git, abra um terminal na pasta que você deseja transformar em um repositório e execute:
```shell
git init
```

### Primeiro commit
Depois de incluir algum conteúdo, você pode salvar o estado do repositório fazendo um commit. Mas antes disso, você deve especificar *quais* arquivos você quer incluir nesse commit! Você não necessariamente vai guardar o diretório inteiro.

Para essa especificação, o git define uma área chamada _staging_. Um arquivo está _staged_ se ele está marcado para ser incluso no próximo commit, mas não foi commitado ainda (portanto, atenção: O conteúdo ainda não foi salvo!)

Para colocar um arquivo na área de staging, execute:
```
git add <nome_do_arquivo>
```

> Esse [tutorial](https://blog.betrybe.com/git/git-add/) detalha mais o comando `git add`

Execute para todos os arquivos que você deseja incluir no próximo commit, ou simplesmente
```shell
git add .
```

, que inclui o diretório todo. 

Para checar quais arquivos estão na área de staging, você pode executar:

```
git status
```

que deve exibir uma sessão chamada `Changes to be committed`, que contém os arquivos staged; e possivelmente as sessões:

- `Changes not staged for commit`, que contém arquivos gerenciados pelo git e que foram modificados desde o último commit, mas não estão staged (e portanto não serão atualizados no próximo commit)

- `Untracked files`, que contém arquivos que nunca foram gerenciados pelo git (i. e., nunca foram "commitados")

Uma vez que você tenha certeza que os arquivos na área de staging são os que você quer commitar, podemos fazer isso. Também é importante incluir uma mensagem que indique o que você alterou nesse commit: Dessa forma, é mais fácil identificar o commit no futuro, caso você precise voltar até ele, por exemplo. Para definir a mensagem, usamos a flag `-m` do comando `git commit`.

```
git commit -m "Adiciona o README e o conteúdo inicial"
```

> Esse [tutorial](https://blog.betrybe.com/git/git-commit/) detalha o comando `git commit`

E pronto! O conteúdo dos arquivos foi salvo, e futuramente você pode restaurá-los pra como eles são hoje.

### Voltando a outro commit

Cada commit tem um _hash_, um código que o identifica. Para restaurar o repositório para o estado de um commit específico, podemos executar:

```shell
git checkout <hash>
```

E o conteúdo do repositório será restaurado para o conteúdo que estava nele no momento do commit com aquele hash. Mas qual é o hash do commit para onde eu quero voltar?

Para descobrir isso, você pode executar

```
git log
```

que vai exibir uma lista dos commits feitos, incluindo a mensagem de cada um deles (!) e o seu hash.

> AVISO: Voltar a versões antigas do código não é o principal uso do comando `git checkout`. Usá-lo assim pode trazer comportamentos inesperados, então a princípio, evite! Na próxima seção veremos o uso principal desse comando.

### Branches

O git organiza o histórico do seu código de forma ramificada; ou seja, você pode "bifurcar" o histórico em determinado ponto e ter dois ou mais históricos (ou sequência de commits) separados. Cada "ramo" é chamado de _branch_, e as alterações feitas em uma branch não interferem nas outras. Isso é útil para o desenvolvimento colaborativo, de forma que o trabalho de um colaborador não interfere no trabalho de outro.

Futuramente, as mudanças feitas em mais de uma branch podem ser integradas em um único commit por meio do _merge_, que significa, na tradução literal, mesclar. Um merge gera um novo commit, e nele, as mudanças feitas em ambas as branchs estarão disponíveis.

Esse [tutorial](https://www.atlassian.com/br/git/tutorials/using-branches) detalha melhor o conceito de branches com ilustrações e uma visão um pouco mais aprofundada.

A branch padrão costuma se chamar `main` ou `master`, dependendo do sistema. Você pode conferir qual é a sua branch atual executando
```
git branch
```

Para criar uma nova branch e mudar para ela, use o comando
```
git checkout -b <nome_da_branch>
```

> OBS: A flag `-b` significa que a branch não existe, e estamos criando ela. Sem essa flag, o comando tentaria navegar para uma branch existente chamada <nome_da_branch>

Depois disso, verifique que de fato está na branch nova executando `git branch` de novo. Se tudo deu certo, os próximos commits serão feitos nessa nova branch, e não na main.

Para navegar entre as branches, use por exemplo:
```
git checkout main
```

Observe que não só isso significa que os próximos commits serão adicionados à main, como o conteúdo atual do repositório vai ser substituído pelo do commit mais recente da `main` (nesse caso). Isso significa que qualquer alteração feita e commitada na sua branch vai ser desfeita (temporariamente, claro. Essas alterações estão salvas e serão restauradas se você executar `git checkout <nome_da_branch>`).

Finalmente, depois de fazer as modificações desejadas na sua branch separada, você pode querer incorporá-las ao código principal (na `main`). Para isso, certifique-se que você está na branch `main`
```
git checkout main
```

E execute 
```
git merge <nome_da_branch>
```

A princípio, vai ser feito um novo commit na main que inclui tudo que foi feito nessa branch, mas pode ocorrer de o mesmo trecho ter sido modificado em ambas as branches. Se isso ocorre, o git não sabe qual versão manter. Isso se chama _conflito de merge_, e você deve resolvê-lo manualmente, especificando qual versão manter.

No github, uma forma mais conveniente de fazer merges é por meio de um _pull request_. [Esse repositório](https://github.com/aprenda-git/pull-request) detalha o processo dos pull requests. Leia ele após a seção de colaboração!

## Colaboração

Para desenvolver em conjunto com outras pessoas, precisamos do que chamamos de _repositório remoto_: um repositório compartilhado por várias pessoas, acessado pela internet. Aqui, vamos usar o github para hospedar nossos repositórios. Para essa tarefa, não precisamos criar um novo repositório remoto, então vamos focar em como utilizar um existente.

### Clonando o repositório com git clone

Para baixar um repositório remoto na sua máquina e poder começar a desenvolver nele, usamos o comando `git clone`:

```shell
git clone <url_do_repositório>
```

E um diretório será criado com o mesmo nome que o repositório.

Esse [tutorial](https://www.alura.com.br/artigos/clonando-repositorio-git-github) detalha mais essa parte do processo, e também dá exemplos de uso de outros comandos que vimos até aqui.

Quando você faz um commit localmente, essas mudanças são salvas apenas na sua máquina! Ou seja, o commit não atualiza automaticamente o conteúdo do repositório remoto, e outros colaboradores vão continuar tendo acesso apenas à versão antiga.

Para enviar suas mudanças (todos os commits feitos) para o repositório remoto, usamos:
```
git push
```

> Esse [tutorial](https://blog.betrybe.com/git/git-push/) detalha o comando `git push`

Similarmente, quando outro colaborador executa o `git push`, seu repositório local **não** é automaticamente atualizado. Para recuperar as mudanças enviadas por outras pessoas, execute:
```
git pull
```

> AVISO: Crie o hábito de sempre executar o `git pull` **antes** de começar a trabalhar/fazer mudanças. Isso evita que você altere coisas que outra pessoa já alterou, causando um conflito. Também é boa prática trabalhar em branches diferentes, para facilitar o tratamento de conflitos no futuro.

