#!/usr/bin/env python
# coding: utf-8

# <span style='font-family:serif'>
#     
# # <center>$Machine   Learning   From   Scratch$</center>
# # <center><span style='background:yellow'>Naive Bayes</span></center>
# <center>$Rafael Pavan$</center>
# 
# 

# <span style='font-family:serif'>
#     
# ## 1. Introdução
# 
# Este notebook irá realizar a formulação matemática do algoritmo Naive Bayes. O Naive-Bayes é um método de aprendizado supervisionado baseado com funcionamento baseado em probabilidade. O algoritmo utiliza o Teorema de Bayes supondo independência total entre os atributos de uma amostra, por isso é denominado ingênuo (Naive). O método se destaca em aplicações de Processamento de Linguagem Natural (PLN).
#     
# Dados:
#  
# Rodonildo é um jogador nato de *League of Legends*, um jogo de estratégia que envolve a batalha entre dois times, e esteve coletando dados nas partidas em que jogou. O objetivo de Rodonildo é prever o vencedor de uma determinada batalha a partir de algumas informações. Na coleta de dados que Rodonildo fez, ele utilizou amostras compostas pelos 5 atributos binários (*1 = sim* e *0 = não*) a seguir:
#     
# 
# 1. *primeiroAbate*: indica se a primeira morte do jogo foi realizada pelo time de Rodonildo;
# 2. *primeiraTorre*: indica se a primeira torre destruída do jogo foi derrubada pelo time de Rodonildo (Figura 1a);
# 3. *primeiroInibidor*: indica se o primeiro inibidor destruído do jogo foi derrubado pelo time de Rodonildo (Figura 1b);
# 4. *primeiroDragao*: indica se o personagem Dragão foi abatido primeiro pelo time de Rodonildo (Figura 1c);
# 4. *primeiroBaron*: indica se o personagem Baron foi abatido primeiro pelo time de Rodonildo (Figura 1d).
# 
#     
# Todos os atributos deste problema de classificação possuem apenas dois valores possíveis (1 \[a ação representada pelo atributo foi tomada pelo time do Rodonildo\] ou 0 [a ação representada pelo atributo foi tomada pelo time adversário]). Portanto, na função abaixo você deverá calcular apenas a probabilidade do atributo possuir valor 1. Posteriormente, na função de classificação, basta considerar que a probabilidade de um determinado atributo possuir valor 0 é complementar à probabilidade do atributo possui valor 1. 

# In[21]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
plt.style.use('fivethirtyeight')


# <span style='font-family:serif'>
#     
# ## 2. Criando e Pré-Visualizando os Dados
# 

# In[22]:


df_dataset = pd.read_csv('dados.csv', sep=',', index_col=None)

df_dataset.head()


# In[23]:


X = df_dataset.iloc[:, 0:-1].values 
Y = df_dataset.iloc[:, -1].values 


# In[24]:


print('X:', X[0:5,:])

print('Y:', Y[0:5])


# <span style='font-family:serif'>
#     
# ## 2. Probabilidade de Ocorrência
#  
# 
# 

# In[25]:


pVitoria = sum(Y==1)/len(Y) 
pDerrota = sum(Y==0)/len(Y)

print('Probabilidade da classe ser 1 : %1.2f%%' %(pVitoria*100))
print('Probabilidade da classe ser 0 : %1.2f%%' %(pDerrota*100))


# In[26]:


def calcula_Probabilidades(X, Y):
    
    """
    
    Calcula a probabilidade de ocorrência de cada atributo por classe possível.
    A funcao retorna dois vetores de tamanho n (número de atributos), um para cada classe.
    
    """
    
    pbVitoria = np.zeros(X.shape[1])
    pbDerrota = np.zeros(X.shape[1])
    
    for atributo in range(X.shape[1]):
        
        filtrov = X[:,atributo] == 1

        Yv = Y[filtrov]

        pbVitoria[atributo] = ((len(Yv[Yv==1]))/len(Yv)*(len(X[X[:,atributo]==1])/len(X[:,atributo])))/(len(Y[Y==1])/len(Y))
        pbDerrota[atributo] = ((len(Yv[Yv==0]))/len(Yv)*(len(X[X[:,atributo]==1])/len(X[:,atributo])))/(len(Y[Y==0])/len(Y))

    return pbVitoria, pbDerrota

pbVitoria, pbDerrota = calcula_Probabilidades(X,Y)

print('A probabilidade esperada para P(PrimeiroAbate=1|Classe=1) = %.2f%%' %52.96)
print('\nEssa mesma probabilidade calculada no seu codigo foi = %.2f%%' %(pbVitoria[0]*100))


# <span style='font-family:serif'>
#     
# ## Parte 2: Classificação da Base Pelo Naive Bayes
# 
# Nesta etapa, é realizada a classificação das amostras com base nas probabilidades encontradas no passo anterior. A classificação é realizada verificando se a amostra em questão tem maior probabilidade de pertencer à classe 1 ou à classe 0. Para calcular a probabilidade de uma amostra pertencer a uma determinada classe, é necessário utilizar as probabilidades de ocorrências de atributos previamente computadas. O cálculo pode ser expresso como:
# 
# $$ P(y_j|\vec{x}) = \hat{P}(y_{j}) \prod_{x_i \in \vec{x}} \hat{P}(x_{i} | y_{j}) $$
# 
# Portanto, a probabilidade de uma amostra $\vec{x}$ pertencer a uma classe $j$ é obtida a partir da probabilidade geral da classe $j$ ($\hat{P}(y_{j})$) multiplicada pelo produtório da probabilidade de ocorrência de cada atributo $x_i$ com relação a classe $j$ ($\hat{P}(x_{i} | y_{j})$).
# 
# Se a rotina de classificação estiver correta, espera-se que a acurácia obtida ao classificar a própria base de amostras de jogos que Ronildo participou seja aproximadamente 76,60%. 
# 

# In[28]:


def classificacao(x,pVitoria,pDerrota,pAtrVitoria,pAtrDerrota):
    
    """
    
    Classifica se a entrada x pertence a classe 0 ou 1 usando as probabilidades extraidas da base de treinamento. 
    Essa função estima a predição de x através da maior probabilidade da amostra pertencer a classe 1 ou 0. 
    Também retorna as probabilidades condicionais de vitoria e derrota, respectivamente.
    
    """
    
    classe = 0
    probVitoria= 0
    probDerrota = 0

    probVitoria = pVitoria
    probDerrota = pDerrota
    
    for j in range(x.shape[0]):
        
        if x[j]== 1: 
            probVitoria=probVitoria*pAtrVitoria[j]
            probDerrota=probDerrota*(pAtrDerrota[j])
        
        else: 
            probVitoria=probVitoria*(1-pAtrVitoria[j])
            probDerrota=probDerrota*(1-pAtrDerrota[j])
    
    if probVitoria > probDerrota:
        classe = 1
    else:
        classe = 0
    
    return classe, probVitoria, probDerrota 

resultados = np.zeros(X.shape[0])

for i in range(X.shape[0]):
    resultados[i], probVitoria, probDerrota = classificacao( X[i,:],pVitoria,pDerrota,pbVitoria,pbDerrota )

acuracia = np.sum(resultados==Y)/len(Y)

print('\n\nAcuracia esperada para essa base = %.2f%%\n' %76.60);
print('Acuracia obtida pelo seu classificador foi = %.2f%%\n' %( acuracia*100 ) )


# <span style='font-family:serif'>
#     
# ## 3. Predição de um Novo Dado

# In[29]:


x1_novo = np.array([0,0,0,1,1])

classe, probVitoria, probDerrota = classificacao( x1_novo,pVitoria,pDerrota,pbVitoria,pbDerrota )

if classe ==1:
    print('\n>>> Predicao = Vitoria!')       
else:
    print('\n>>> Predicao = Derrota!')

print('\n>>>>>> Prob. vitoria = %0.6f!' %(probVitoria))
print('\n>>>>>> Prob. derrota = %0.6f!\n\n'  %(probDerrota))


# <span style='font-family:serif'>
#     
# ## 4. Classificação de SPAM
#     
# Nesta parte do exercício, usaremos o Naive Bayes para classificar SMS spam.
# 
# Veja alguns exemplos de SMS legítimos:
#  * ```Is that seriously how you spell his name?```
#  * ```What you thinked about me. First time you saw me in class.```
#  * ```Ok lar i double check wif da hair dresser already he said wun cut v short. He said will cut until i look nice.```
#  
# Agora veja alguns exemplos de SMS spam:
#  * ```WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.```
#  * ```Thanks for your subscription to Ringtone UK your mobile will be charged £5/month Please confirm by replying YES or NO. If you reply NO you will not be charged.```
#  * ```Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed & Free entry 2 100 wkly draw txt MUSIC to 87066 TnCs http://www.Ldew.com1win150ppmx3age16.```
#     
#     
# Antes de fazer qualquer tarefa de classificação com textos, é importante fazer um pré-processamento para obter melhor resultado na predição. Na função abaixo, os seguintes pré-processamentos são realizados:
# 
#  - deixar todas as palavras com letras minúsculas
#  - substituir os números pela palavra *number*
#  - substituir todas as URLS pela palavra *enderecoweb*
#  - substiuir todos os emails pela palavra *enderecoemail*
#  - substituir o símbolo de dólar pela palavra *dolar*
#  - substituit todos os caracteres não-alfanuméricos por um espaço em branco
#  
# Por fim, também é recomendado eliminar todas as palavras muito curtas. Vamos eliminar qualquer palavra de apenas 1 caracter. 

# In[33]:


import re #regular expression

def preprocessing(text):
    
    # Lower case
    text = text.lower()
    
    # remove tags HTML
    regex = re.compile('<[^<>]+>')
    text = re.sub(regex, " ", text) 

    # normaliza os numeros 
    regex = re.compile('[0-9]+')
    text = re.sub(regex, "number", text)
    
    # normaliza as URLs
    regex = re.compile('(http|https)://[^\s]*')
    text = re.sub(regex, "enderecoweb", text)

    # normaliza emails
    regex = re.compile('[^\s]+@[^\s]+')
    text = re.sub(regex, "enderecoemail", text)
    
    #normaliza o símbolo de dólar
    regex = re.compile('[$]+')
    text = re.sub(regex, "dolar", text)
    
    # converte todos os caracteres não-alfanuméricos em espaço
    regex = re.compile('[^A-Za-z]') 
    text = re.sub(regex, " ", text)
    
    # substitui varios espaçamentos seguidos em um só
    text = ' '.join(text.split())
        
    return text

smsContent = 'Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed & Free entry 2 100 wkly draw txt MUSIC to 87066 TnCs http://www.Ldew.com1win150ppmx3age16.'
print('Antes do preprocessamento: \n\n', smsContent)

# chama a função de pré-processsamento para tratar o SMS
smsContent = preprocessing(smsContent)

print('\nDepois do preprocessamento: \n\n', smsContent)


# <span style='font-family:serif'>
#     
# 
# Depois de fazer o pré-processamento, é necessário transformar o texto em um vetor de atributos com valores numéricos. Uma das formas de fazer isso é considerar que cada palavra da base de dados de treinamento é um atributo, cujo valor é o número de vezes que ela aparece em uma determinada mensagem.
# 
# Para facilitar, já existe um vocabulário no arquivo *vocab* que foi previamente extraído. Cada palavra desse vocabulário será considerado um atributo do problema de classificação de spam.
# 
# O código abaixo carrega o vocabulário.

# In[37]:


vocabulario = []

with open('vocab.txt', 'r') as f:
    for line in f:
        line = line.replace('\n','')
        
        vocabulario.append(line)

print('50 primeiras palavras do vocabulário:\n')
print(vocabulario[0:50])


# In[38]:


def text2features(text, vocabulario):
   
    """
    Converte um texto para um vetor de atributos
    """
    
    textVec = np.zeros( [1,len(vocabulario)], dtype=int )
    
    tokens = text.split() # separa as palavras com base nos espaços em branco
    
    tokens = [w for w in tokens if len(w)>1]

    for w in range(len(vocabulario)):
    
        textVec[0,w] =text.count(vocabulario[w])
    
    return textVec

smsVec = text2features(smsContent, vocabulario)

print('Vetor de features correspondente ao SMS:')
print(smsVec[0:50])


# In[39]:


# Importa o arquivo numpy

dataset4_train = np.load('spamData.npz')['train']
dataset4_test = np.load('spamData.npz')['test']

# pega os valores das n-1 primeiras colunas e guarda em uma matrix X

X4_train = dataset4_train[:, 0:-1]
X4_test = dataset4_test[:, 0:-1]

# pega os valores da última coluna e guarda em um vetor Y

Y4_train = dataset4_train[:, -1] 
Y4_test = dataset4_test[:, -1] 

display('X_train:', X4_train[0:5,:])
display('X_test:', X4_test[0:5,:])


print('Y_train:', Y4_train[0:5])
print('Y_test:', Y4_test[0:5])


# In[40]:


# Probabilidade das Classes

pSpam = sum(Y4_train==1)/len(Y4_train) 
pHam = sum(Y4_train==0)/len(Y4_train)

print('Probabilidade da classe ser 1 (spam): %1.2f%%' %(pSpam*100))
print('Probabilidade da classe ser 0 (ham): %1.2f%%' %(pHam*100))


# Para calcular as probabilidades de ocorrência de cada atributo em cada classe irá ser usada a fórmula com correção de Laplace no cálculo da probabilidade.
# 
# $$\hat{P}(w_i|c)=\frac{count(w_i|c)+1}{count(c)+|V|},$$
# 
# onde $w_i$ é um termo do vocabulário, $count(c)$ é quantidade de termos nas amostras da classe $c$ e $|V|$ é o tamanho do vocabulário (número de atributos).

# In[47]:


def calcularProbabilidades_Laplace(X, Y):
    
    """
    
    CALCULARPROBABILIDADES Computa a probabilidade de ocorrencia de cada 
    atributo por rotulo possivel. A funcao retorna dois vetores de tamanho n
    (qtde de atributos), um para cada classe.
    
    CALCULARPROBABILIDADES(X, Y) calcula a probabilidade de ocorrencia de cada atributo em cada classe. 
    Cada vetor de saida tem dimensao (n x 1), sendo n a quantidade de atributos por amostra.
    
    """
    
    pAtrSpam = np.zeros(X.shape[1])
    pAtrHam = np.zeros(X.shape[1])

    filtrov1 = Y == 1
    Xv1 = X[filtrov1]  
    filtrod1 = Y == 0
    Xd1 = X[filtrod1]  
    
    for atribut in range(X.shape[1]):
        pAtrSpam[atribut] = (np.sum(Xv1[:,atribut],axis=0)+1)/(len(Y[Y==1]) + X.shape[1])
        
        pAtrHam[atribut] = (np.sum(Xd1[:,atribut],axis=0)+1)/(len(Y[Y==0]) + X.shape[1])
    


    return pAtrSpam, pAtrHam

pAtrSpam, pAtrHam = calcularProbabilidades_Laplace(X4_train,Y4_train)

print('\nA probabilidade calculada no código foi = %.8f' %(pAtrSpam[0]))


# <span style='font-family:serif'>
#     
# Agora, vamos realizar a classificação das amostras com base nas probabilidades encontradas no passo anterior. A classificação é realizada verificando se a amostra em questão tem maior probabilidade de pertencer à classe 1 ou à classe 0. Conforme vimos no exercício anterior, para calcular a probabilidade de uma amostra pertencer a uma determinada classe, é necessário utilizar as probabilidades de ocorrências de atributos previamente computadas:
# 
# $$ P(y_j|\vec{x}) = \hat{P}(y_{j}) \prod_{x_i \in \vec{x}} \hat{P}(x_{i} | y_{j}) $$
# 
# Em classificação de textos, a probabilidade de ocorrência de cada termo geralmente é muito próxima de 0. Quando você multiplica essas probabilidades, o resultado final se aproxima ainda mais de 0, o que pode causar estouro de precisão numérica.
# 
# Um truque para evitar esse problema é substituir a equação acima por:
# 
# $$ P(y_j|\vec{x}) = \log\left(\hat{P}(y_{j})\right) + \sum_{x_i \in \vec{x}} \log\left(\hat{P}(x_{i} | y_{j})\right) $$
# 

# In[48]:


def classificacao_texto(x,pSpam,pHam,pAtrSpam,pAtrHam):
    """
    Classifica se a entrada x pertence a classe 0 ou 1 usando
    as probabilidades extraidas da base de treinamento. Essa funcao 
    estima a predicao de x atraves da maior probabilidade da amostra  
    pertencer a classe 1 ou 0. Tambem retorna as probabilidades condicionais
    de vitoria e derrota, respectivamente.
    
    """

    classe = 0;
    probSpam = 0;
    probHam = 0;

    pAtrSpam = np.log(pAtrSpam)
    pAtrHam = np.log(pAtrHam)
    
    probSpam = np.log(pSpam) + probSpam+np.sum(pAtrSpam[np.where(x!=0)])
    probHam = np.log(pHam) + probHam+np.sum(pAtrHam[np.where(x!=0)])
    
    if probSpam  > probHam :
        classe=1
    else:
        classe=0


    return classe, probSpam, probHam   
    

resultados = np.zeros( X4_test.shape[0] )
for i in range(X4_test.shape[0]):
    resultados[i], probSpam, probHam = classificacao_texto( X4_test[i,:],pSpam,pHam,pAtrSpam,pAtrHam )

# calcular acuracia
acuracia = np.sum(resultados==Y4_test)/len(Y4_test)

print('Acuracia obtida pelo classificador foi = %.2f%%\n' %( acuracia*100 ) )


# In[55]:


smsContent = 'Congratulations ur awarded 500 of CD vouchers or 125gift guaranteed & Free entry 2 100 wkly draw txt MUSIC to 87066 TnCs http://www.Ldew.com1win150ppmx3age16.'

smsContent2 = 'Hello, how are you doing? The results of you application are available already! You can see your score in the link: http://www.linkscore.com.br'

print(smsContent) 

# chama a função de pré-processsamento para tratar o email
smsContent = preprocessing(smsContent)
smsContent2 = preprocessing(smsContent2)

# converte o texto para um vetor de features
smsVec = text2features(smsContent, vocabulario)
smsVec2 = text2features(smsContent2, vocabulario)

# classifica o email
classe, probSpam, probHam = classificacao_texto( smsVec[0,:],pSpam,pHam,pAtrSpam,pAtrHam )

if classe==1:
    print('\n>>> Predicao = Spam!')       
else:
    print('\n>>> Predicao = Ham!')


# In[56]:


# classifica o email
classe, probSpam, probHam = classificacao_texto( smsVec2[0,:],pSpam,pHam,pAtrSpam,pAtrHam )


print(smsContent2) 


if classe==1:
    print('\n>>> Predicao = Spam!')       
else:
    print('\n>>> Predicao = Ham!')

