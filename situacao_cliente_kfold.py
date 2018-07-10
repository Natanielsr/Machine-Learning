from collections import Counter
from sklearn.cross_validation import cross_val_score
import pandas as pd
import numpy as np


# pandas importando data frame d. f.
df = pd.read_csv('situacao_cliente.csv')

#X são os dados que ficam do lado esquerdo da minha tabela
X_df = df[['recencia', 'frequencia', 'semanas_de_inscricao']]

# Y é as marcações, o resultado do lado direito da minha tabela
Y_df = df['situacao']

#traduz os valores das categorias em valores binarios
Xdummies_df = pd.get_dummies(X_df)

#se utilizar o dummies no Y devolve 2 colunas
# como Y ja esta em binario não ha necessidade de converter
# so seria necessario o dummies se o valor fosse algo como 'SIM' ou 'NAO'
Ydummies_df = Y_df

#converte o data frame em array
X = Xdummies_df.values
Y = Ydummies_df.values



#caclula a quantidade de 80% dos valores
porcentagem_treino = 0.8
tamanho_de_treino = int(porcentagem_treino * len(X))
#cria um array com 80% dos dados
treino_dados = X[0:tamanho_de_treino]
treino_marcacoes = Y[0:tamanho_de_treino]


#calcula a quantidade de validacao
tamanho_de_validacao = len(Y) - tamanho_de_treino
validacao_dados = X[tamanho_de_treino:]
validacao_marcacoes = Y[tamanho_de_treino:]





def fit_and_predict(modelo, treino_dados, treino_marcacoes):
    k = 10
    scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv = k )

    taxa_de_acerto = np.mean(scores)

    msg = "Taxa de acerto do {0}: {1}".format(modelo.__class__.__name__, taxa_de_acerto)

    print(msg)

    return taxa_de_acerto


resultados = {}

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modeloOneVsRest = OneVsRestClassifier(LinearSVC(random_state=0))
resultadoOneVsRest = fit_and_predict(modeloOneVsRest, treino_dados, treino_marcacoes)
resultados[resultadoOneVsRest] = modeloOneVsRest

from sklearn.multiclass import OneVsOneClassifier
modeloOneVsOne = OneVsOneClassifier(LinearSVC(random_state=0))
resultadoOneVsOne = fit_and_predict(modeloOneVsOne, treino_dados, treino_marcacoes)
resultados[resultadoOneVsOne] = modeloOneVsOne


#treino com algoritmo multinomialnb
from sklearn.naive_bayes import MultinomialNB
modeloMultinomialNB = MultinomialNB()
resultadoMultinomialNB = fit_and_predict(modeloMultinomialNB, treino_dados, treino_marcacoes)
resultados[resultadoMultinomialNB] = modeloMultinomialNB



#treina com adaboost
from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoostClassifier = AdaBoostClassifier()
resultadoAdaBoostClassifier = fit_and_predict(modeloAdaBoostClassifier, treino_dados, treino_marcacoes)

resultados[resultadoAdaBoostClassifier] = modeloAdaBoostClassifier


maximo = max(resultados)
vencedor = resultados[maximo]

vencedor.fit(treino_dados, treino_marcacoes)

#valida o algoritmo vencedor
resultado = vencedor.predict(validacao_dados)
acertos = (resultado == validacao_marcacoes)

total_de_acertos = sum(acertos)
total_de_elementos = len(validacao_marcacoes)
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

msg = "Taxa de acerto vencedor {}: {}%".format(vencedor.__class__.__name__, taxa_de_acerto)

print(msg)

print('Quantidade de Elementos: {}'.format(len(Y)))
print('Quantidade de treino: {}'.format(tamanho_de_treino))
print('Quantidade de validacao: {}'.format(tamanho_de_validacao))




#a eficacia do algoritmo que chuta tudo 0 ou 1

#pega a quantidade do valor que mais aparece
#Counter conta a quantidade de cada valor e os separa Counter({1: 4, 0: 1})
#values pega apenas a quantidade dos valores [4, 1]
# iter itera sobre os valores
#max pega o maior numero
acerto_base =  max(iter(Counter(validacao_marcacoes).values()))

taxa_de_acerto_base = acerto_base / len(validacao_marcacoes) * 100.0

print('Taxa de acerto base: {}%'.format(taxa_de_acerto_base))

#--------------------------------------------------------------