from collections import Counter
import pandas as pd

# pandas importando data frame d. f.
df = pd.read_csv('buscas.csv')

#X são os dados que ficam do lado esquerdo da minha tabela
X_df = df[['home', 'busca', 'logado']]

# Y é as marcações, o resultado do lado direito da minha tabela
Y_df = df['comprou']

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



#calcula a quantidade de valores dos 10% que sobraram
porcentagem_teste = 0.1
tamanho_de_teste =  int(porcentagem_teste * len(X))
#cria um array com 10% dos dados
fim_teste = tamanho_de_treino + tamanho_de_teste
teste_dados = X[tamanho_de_treino: fim_teste]
teste_marcacoes = Y[tamanho_de_treino: fim_teste]



#calcula a quantidade de validacao
tamanho_de_validacao = len(Y) - tamanho_de_treino - tamanho_de_teste
fim_validacao = fim_teste + tamanho_de_validacao
validacao_dados = X[fim_teste: fim_validacao]
validacao_marcacoes = Y[fim_teste: fim_validacao]



def fit_and_predict(nome, modelo):
      
   #treina o modelo
   modelo.fit(treino_dados, treino_marcacoes)
   return predict(nome, modelo, teste_dados, teste_marcacoes)
  

def predict(nome, modelo, dados, marcacoes):
    #testa o modelo
   resultado = modelo.predict(dados)

   #cria um array de true e false
   acertos = (resultado == marcacoes)

   #true tem valor 1, entao soma os 1
   quantidade_de_acertos = sum(acertos)

   # 10 /100 = 0.10 * 100 = 10%
   taxa_de_acerto = quantidade_de_acertos / tamanho_de_teste * 100
   print('Quantidade de acertos do algoritmo {} é {}'.format(nome, quantidade_de_acertos))  
   print('Taxa de acerto do algoritmo {} é {}%'.format(nome, taxa_de_acerto))

   return taxa_de_acerto

#treino com algoritmo multinomialnb
from sklearn.naive_bayes import MultinomialNB
modeloMultinomialNB = MultinomialNB()
resultadoMultinomialNB = fit_and_predict('MultinomialNB', modeloMultinomialNB)

print('\n')

#treina com adaboost
from sklearn.ensemble import AdaBoostClassifier
modeloAdaBoostClassifier = AdaBoostClassifier()
resultadoAdaBoostClassifier = fit_and_predict('AdaBoostClassifier', modeloAdaBoostClassifier)

print('\n')

#verifica qual algoritmo teve a maior taxa de acerto
if(resultadoMultinomialNB > resultadoAdaBoostClassifier):
   vencedor = modeloMultinomialNB
else:
   vencedor = modeloAdaBoostClassifier

#valida o algoritmo vencedor
predict('Validacao '+vencedor.__class__.__name__ ,vencedor, validacao_dados, validacao_marcacoes)

print('\n')

print('Quantidade de treino: {}'.format(tamanho_de_treino))
print('Quantidade de testes: {}'.format(tamanho_de_teste))
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