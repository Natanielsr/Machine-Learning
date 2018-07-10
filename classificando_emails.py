#!-*- conding: utf8 -*-
from collections import Counter
from sklearn.cross_validation import cross_val_score
import numpy as np


porcentagem_treino = 0.8

texto1 = "Se eu comprar cinco anos antecipados, eu ganho algum desconto?"
texto2 = "O exercício 15 do curso de Java 1 está com a reposta errada"
texto3 = "Existe algum curso para cuidar do marketing da minha empresa?"

import pandas as pd

def quebrarTexto(classificacoes, colunaTexto):
   textosPuros = classificacoes[colunaTexto]
   textosQuebrados = textosPuros.str.lower().str.split(' ')

   return textosQuebrados

def montaDicionario(textosQuebrados):

   dicionario = set()

   for frase in textosQuebrados:
      dicionario.update(frase)

   return dicionario

def montaTradutor(dicionario):

   totalDePalavras = len(dicionario)
   tuplas = zip(dicionario, range(totalDePalavras))

   tradutor = {palavra:indice for palavra, indice in tuplas}

   return tradutor

def montaVetores(tradutor, textosQuebrados):
   return [vetorizar_texto(texto, tradutor) for texto in textosQuebrados]

def vetorizar_texto(texto, tradutor):

   vetor = [0] * len(tradutor)

   for palavra in texto:
      if palavra in tradutor:
         posicao = tradutor[palavra]
         vetor[posicao] += 1

   return vetor



def divideDados(X, Y, porcentagem_de_treino):

   tamanho_do_treino = int(porcentagem_de_treino * len(Y))
   tamanho_de_validacao = len(Y) - tamanho_do_treino

   treino_dados = X[0:tamanho_do_treino]
   treino_marcacoes = Y[0:tamanho_do_treino]

   validacao_dados = X[tamanho_do_treino:]
   validacao_marcacoes = Y[tamanho_do_treino:]

   mostrar_specs(Y, tamanho_do_treino, tamanho_de_validacao)

   return treino_dados, treino_marcacoes, validacao_dados, validacao_marcacoes


def fit_and_predict(modelo, treino_dados, treino_marcacoes):
    k = 10
    scores = cross_val_score(modelo, treino_dados, treino_marcacoes, cv = k )

    taxa_de_acerto = 100.0 * np.mean(scores)

    msg = "Taxa de acerto do {0}: {1}%".format(modelo.__class__.__name__, taxa_de_acerto)

    print(msg)

    return taxa_de_acerto

def testaModelo(resultados, modelo, treino_dados, treino_marcacoes):
   resultado = fit_and_predict(modelo, treino_dados, treino_marcacoes)
   resultados[resultado] = modelo

def classificaVencedor(resultados, treino_dados, treino_marcacoes, validacao_dados, validacao_marcacoes):
    maximo = max(resultados)
    vencedor = resultados[maximo]

    vencedor.fit(treino_dados, treino_marcacoes)

    #valida o algoritmo vencedor
    resultado = vencedor.predict(validacao_dados)
    acertos = (resultado == validacao_marcacoes)

    total_de_acertos = sum(acertos)
    total_de_elementos = len(validacao_marcacoes)

    print("Quantidade de acertos do {}: {} de {}".format(vencedor.__class__.__name__, total_de_acertos, total_de_elementos))

    taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

    msg = "Taxa de acerto vencedor {}: {}%".format(vencedor.__class__.__name__, taxa_de_acerto)
    print(msg)

def taxa_acerto_base(validacao_marcacoes):
    #a eficacia do algoritmo que chuta tudo 0 ou 1

    #pega a quantidade do valor que mais aparece
    #Counter conta a quantidade de cada valor e os separa Counter({1: 4, 0: 1})
    #values pega apenas a quantidade dos valores [4, 1]
    # iter itera sobre os valores
    #max pega o maior numero
    acerto_base =  max(iter(Counter(validacao_marcacoes).values()))

    taxa_de_acerto_base = acerto_base / len(validacao_marcacoes) * 100.0

    print('Taxa de acerto base: {}%'.format(taxa_de_acerto_base))

def mostrar_specs(Y, tamanho_de_treino, tamanho_de_validacao):
    print('Quantidade de Elementos: {}'.format(len(Y)))
    print('Quantidade de treino: {}'.format(tamanho_de_treino))
    print('Quantidade de validacao: {}'.format(tamanho_de_validacao))


classificacoes = pd.read_csv('emails.csv')
textosQuebrados = quebrarTexto(classificacoes, 'email')

dicionario = montaDicionario(textosQuebrados)

tradutor = montaTradutor(dicionario)

print("Tamanho Dicionario ",len(dicionario))

vetores = montaVetores(tradutor, textosQuebrados)
marcas = classificacoes['classificacao']

treino_dados, treino_marcacoes, validacao_dados, validacao_marcacoes = divideDados(vetores, marcas, porcentagem_treino)

modelos = []

from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
modelo = OneVsRestClassifier(LinearSVC(random_state=0))
modelos.append(modelo)

from sklearn.multiclass import OneVsOneClassifier
modelo = OneVsOneClassifier(LinearSVC(random_state=0))
modelos.append(modelo)

#treino com algoritmo multinomialnb
from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()
modelos.append(modelo)

#treina com adaboost
from sklearn.ensemble import AdaBoostClassifier
modelo = AdaBoostClassifier()
modelos.append(modelo)


resultados = {}
for m in modelos:
   testaModelo(resultados, m, treino_dados, treino_marcacoes)

classificaVencedor(resultados, treino_dados, treino_marcacoes, validacao_dados, validacao_marcacoes)

taxa_acerto_base(validacao_marcacoes)

#print(resultados)