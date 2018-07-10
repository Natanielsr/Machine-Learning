#é gordinho? tem perninha curta? se faz auau

porco1 = [1, 1, 0]
porco2 = [1, 1, 0]
porco3 = [1, 1, 0]
cachorro1 = [1, 1, 1]
cachorro2 = [0, 1, 1]
cachorro3 = [0, 1, 1]

dados = [
   porco1,
   porco2,
   porco3,
   cachorro1,
   cachorro2,
   cachorro3
]

#-1 cachorro 1 porco
marcacoes = [1, 1, 1, -1, -1, -1]

#importa algoritmo
from sklearn.naive_bayes import MultinomialNB

#treina
modelo = MultinomialNB()
modelo.fit(dados, marcacoes)


misterioso1 = [1, 1, 1]
misterioso2 = [1, 0, 0]
misterioso3 = [0, 0, 1]

testes = [misterioso1, misterioso2, misterioso3]

#-1 cachorro 1 porco
marcacoes_teste = [-1, 1, -1]

#preveja
chute = modelo.predict(testes)

diferencas = chute - marcacoes_teste

print('Previsão do algoritmo {}'.format(chute))
print('Respostas corretas    {}'.format(marcacoes_teste))

acertos = [d for d in diferencas if d == 0]

total_de_acertos = len(acertos)
total_de_elementos = len(testes)

taxa_de_acerto = total_de_acertos / total_de_elementos * 100

print('Taxa de acerto {}%'.format(taxa_de_acerto))