from dados import carregar_acessos

X, Y = carregar_acessos()

print('Numero de itens da amostra {}'.format(len(Y)))

#pega 90% dos dados e marcacoes para o treino do modelo
treino_dados = X[:90]
treino_marcacoes = Y[:90]

print('Numero de itens para treino {}'.format(len(treino_marcacoes)))

#pega 10% dos dados e marcacoes para teste do modelo
teste_dados = X[-9:]
teste_marcacoes = Y[-9:]

print('Numero de itens para teste {}'.format(len(teste_marcacoes)))

from sklearn.naive_bayes import MultinomialNB

modelo = MultinomialNB()

#treina o modelo com os arrays de treino de 90% que foram separados
modelo.fit(treino_dados, treino_marcacoes)

#preveja o resultado com os arrays de 10% de dados teste que foram separados
resultado = modelo.predict(teste_dados)

#calcular a diferenca, se for 0 acertou,  1 - 1 = 0, 0 - 0 = 0
#testa com array de 10% de marcacoes que foi separado
diferencas = resultado - teste_marcacoes

#cria um array com a diferenca que for 0
acertos = [d for d in diferencas if d == 0]

#calcula o tamanho do array de acertos (quantidade de acertos)
total_de_acertos = len(acertos)

print('numero de acertos {}'.format(total_de_acertos))

#cacula o total do array dos resultados reais (total dos resultados reais)
total_de_elementos = len(teste_marcacoes)

#ex.  1 / 10 = 0.10 * 100 = 10.0%
taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print('numero de elementos {}'.format(total_de_elementos))
print('taxa de acerto {}%'.format(taxa_de_acerto))