# Machine-Learning
Notebooks utilizando Python 3 com alguns algoritmos de Machine Learning feitos durante a disciplina de Aprendizagem de Máquina.

## Descrição

### Q1_SVM
Considere o problema de classificação de padrões constituído de duas classes com os
seguintes conjunto de treinamentos: <br />
C<sub>1</sub> = {(0,0,0), (1,0,0), (1,0,1), (1,1,0)} e C<sub>2</sub> = {(0,0,1), (0,1,1), (0,1,0), (1,1,1)}. Determine o 
hiperplano de separação dos padrões usando SVM com kernel linear.

### Q2_SOM_Final
A propriedade de ordenação topológica do algoritmo SOM pode ser usada para formar
uma representação bidimensional abstrata de um espaço de entrada de alta
dimensionalidade. Para investigar esta forma de representação, considere uma grade
bidimensional consistindo de 10x10 neurônios que é treinada tendo como entrada os dados
oriundos de quatro distribuições gaussianas, C<sub>1</sub>, C<sub>2</sub>, C<sub>3</sub>, e C<sub>4</sub>, em um espaço de entrada de
dimensionalidade igual a oito, isto é **x** = (x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>8</sub>)<sup>t</sup>. Todas as nuvens têm variâncias
unitária, mas centros ou vetores média diferentes dados por
**m<sub>1</sub>** = (0,0,0,0,0,0,0,0)<sup>t</sup>, **m<sub>2</sub>** = (4,0,0,0,0,0,0,0)<sup>t</sup>, **m<sub>3</sub>** = (0,0,0,4,0,0,0,0)<sup>t</sup>,
**m<sub>4</sub>** = (0,0,0,0,0,0,0,4)<sup>t</sup>.
Calcule o mapa produzido pelo algoritmo SOM, com cada neurônio do mapa sendo
rotulado com a classe particular mais representada pelos pontos de entrada em sua volta. O
objetivo é visualizar os dados de dimensão 8 em um espaço de dimensão 2, constiuido pela
grade de neurônios.

### Q3_SVM
Considere o problema de classificação de padrões bidimensionais constituído neste
caso de 5 padrões. A distribuição dos padrões tem como base um quadrado centrado na
origem interceptando os eixos nos pontos +1 e -1 de cada eixo. Os pontos +1 e -1 de cada
eixo são centros de quatro semicírculos que se interceptam no interior do quadrado originando
uma classe e a outra classe corresponde as regiões de não interseção. Após gerar
aleatoriamente os dados que venham formar estas distribuições de dados, selecione um conjunto
de treinamento e um conjunto de validação. Solucionar este problema utilizando SVM.

Vale frizar que foram testados vários kernels e diferentes valores de **C** (penalização pelo erro).

### Q5_SVM
Um problema interessante para testar a capacidade de uma rede neural atuar como classificado
de padrões é o problema das duas espirais intercaladas. A espiral 1 sendo uma classe
e a espiral 2 sendo outra classe. Gere os exemplos de treinamento usando as seguintes
equações: <br />
para espiral 1: x = <sup>&theta;</sup>&frasl;<sub>4</sub> cos(&theta;) e y = <sup>&theta;</sup>&frasl;<sub>4</sub> sen(&theta;)<br />
para espiral 2: x = (<sup>&theta;</sup>&frasl;<sub>4</sub> + 0.8) cos(&theta;) e y = (<sup>&theta;</sup>&frasl;<sub>4</sub> + 0.8) sen(&theta;)
para &theta; &#8805; 0<br />
fazendo &theta; assumir 100 igualmente espaçados valores entre 0 e 20 radianos. Solucionar este problema utilizando SVM:

### Q6_PCA_Final
Considere o problema de análise de componentes principais (PCA), isto é, determinar
em uma distribuição de dados as componentes que tenham associadas a elas a maior
variância e representar as mesmas no espaço de dados formado pelos autovetores da matriz
de correlação. Neste sentido considere o seguinte problema.
O arquivo de texto 'Amostras_solo.txt' apresenta os dados relativos a amostras de solo. Para cada amostra, tem-se
as medidas das porcentagens de areia (X1), sedimentos (X2), argila (X3) e a quantidade de
material orgânico (X4). Da referido arquivo de texto obtenha as estatísticas descritivas de cada
variável, isto é, a média, a mediana, o desvio padrão, os valores máximo e mínimo. Sob
estas condições:

a) Obtenha desta tabela a matriz de covariância.

b) Desta matriz determine os autovalores ordenados do máximo ao mínimo e os
autovetores correspondentes.

c) Apresente as equações da componentes principais.

d) Calcule os percentuais de variância para cada componente e ordene a classificação das
variáveis segundo este critério.

### Trabalho_Final
Notebook intitulado 'Atenuação de Ruído em Imagens Utilizando Stacked Autoencoders'. Esse notebook usa como base o código 'mnist_denoising.py'
