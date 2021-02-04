#!/usr/bin/env python
# coding: utf-8

# <span style='font-family:serif'>
#     
# # <center>$Machine   Learning   From   Scratch$</center>
# # <center><span style='background:yellow'>Regressão Logística</span></center>
# <center>$Rafael Pavan$</center>
# 
# 

# <span style='font-family:serif'>
#     
# ## 1. Introdução
# 
# 

# A Regressão Logı́stica é um método de aprendizado supervisionado baseado em otimização, que tem como objetivo calcular a probabilidade de uma amostra pertencer a uma determinada classe, modelando o resultado através da função sigmoidal. É uma técnica tradicional de classificação que serve como base para outros métodos mais avançados, como as redes neurais e as máquinas de vetores de suporte.
# 

# In[192]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from random import randrange
rcParams['font.family'] = 'sans-serif'


# <span style='font-family:serif'>
#     
# ## 2. Importando e Pré-Visualizando os Dados
#     
# Os dados representam a classificação de 2 espécies de plantas com base na largura e comprimento de pétala.
# 

# In[193]:


df = pd.read_csv( 'dados.csv', sep=',')


# In[194]:


df.head()


# In[195]:


X = df.iloc[:, 0:-1].values

Y = df.iloc[:, -1].values #TARGET

print('X:', X[0:5,:])

print('Y:', Y[0:5])


# Agora, vamos criar uma função para plotar os dados.

# In[196]:


def visualizaDados(X,Y):

    """
    Função usada para plotar os dados.
    """
    plt.figure(figsize=(16,10))
    
    plt.scatter( X[Y==0,1], X[Y==0,0], label='A', marker='o', color='red', s=80) 
    plt.scatter( X[Y==1,1], X[Y==1,0], label='B', marker='+', color='blue', s=80) 
    plt.ylabel('Largura',fontsize='medium') 
    plt.xlabel('Comprimento',fontsize='medium') 
    plt.title("Espécies", fontsize='x-large')
    plt.grid()
    
visualizaDados(X,Y)

plt.show()


# <span style='font-family:serif'>
#     
# ## 2. Função Sigmoidal
# 
# A função sigmoide é uma função matemática de amplo uso em campos como a economia e a computação. Ela pode ser calculada por:
#     
# $$ g(z) = \frac{1}{1 + e^{-z}}. $$

# In[197]:


def sigmoid(z):
    
    """
    Calcula a funcao sigmoidal  
    """
    
    if isinstance(z, int):
        
        g = 0
    
    else:
    
        g = np.zeros( z.shape );
     
    g = 1/(1+np.exp(-z))
    
    return g


# <span style='font-family:serif'>
#     
# ## 3. Cálculo da Hipótese (Previsão)
# 
# A hipótese $h_\theta(x)$ pode ser calculada pela expressão:
# 
# $$ h_\theta(x) = g \left(\theta^T x \right), $$

# In[198]:


m, n = X.shape 

X = np.column_stack( (np.ones(m),X) )

theta = np.ones(n+1) 


# In[199]:


def hipotese(X,theta):
            
    """
    
    Calcula a hipótese
    
    """
    
    hip = np.zeros(theta.shape)
    
    hip = sigmoid(np.dot(X, theta))
    
    return hip


# In[200]:


hipotese(X,theta)


# <span style='font-family:serif'>
#     
# ## 4. Cálculo da Função de Custo
# 
# A função de custo, por sua vez, pode ser expressa por:
#     
# $$ J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left[-y^{(i)} \log\left(h_\theta(x^{(i)})\right) - \left(1 - y^{(i)}\right) \log\left(1 - h_\theta(x^{(i)})\right) \right], $$
#     
# E o gradiente desta funçao pode ser dado por:
#     
# $$\frac{\partial J(\theta)}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^{m} \left(h_\theta(x^{(i)}) - y^{(i)} \right).x_{j}^{(i)}.$$
# 

# In[201]:


def funcaoCustoEGrad(theta, X, Y):
    
    """
    
    Calcula o custo de usar theta como parametro da regressao logistica 
    para ajustar os dados de X e Y. 
    
    """
    
    m = len(Y) 
    
    J = 0;
    
    grad = np.zeros( len(theta) );
    
    eps = 1e-19
    
    n = X.shape[1] 
    
    J = (1 / m) * np.sum(-Y * np.log(eps+hipotese(X, theta)) - (1 - Y) * np.log(1 - hipotese(X, theta) + eps))
    
    for i in range(n):
        grad[i] = (1/m) * np.dot(X[:,i].T, hipotese(X, theta) - Y)
     
    return J,grad


# In[202]:


funcaoCustoEGrad(theta, X, Y)


# <span style='font-family:serif'>
#     
# ## 5. Otimizando com o Scipy
# 
# Para utilização do Scipy, no método abaixo, deve-se passar a função de custo bem como a matriz jacobiana para se realizar a otimização. Como nossa expressão de custo é única, o nosso gradiente é equivalente ao jacobiano.

# In[203]:


import scipy.optimize  

MaxIter = 500

theta = np.zeros(n+1)

# minimiza a funcao de custo
result = scipy.optimize.minimize(fun=funcaoCusto, x0=theta, args=(X, Y),  
                method='Newton-CG', jac=True, options={'maxiter': MaxIter, 'disp':True})

# coleta os thetas retornados pela função de minimização
theta = result.x

custo, grad = funcaoCusto(theta, X, Y) 

print('\nCusto encontrado: %f\n' %custo)


# In[204]:


result


# In[205]:


x_reg=np.array( [np.min(X[:,1])-2,  np.max(X[:,1])+2] )

y_reg=theta[0]+theta[1]*X[:,1] + theta[2]*X[:,2]

y_reg


# <span style='font-family:serif'>
#     
# ## 6. Visualizando os Dados
# 
# Como poderemos ver, a froteira de separação para este caso era linear.

# In[206]:


visualizaDados(X[:,1:],Y)


plot_x = np.array( [np.min(X[:,1]-2),  np.max(X[:,1]-2)] )
plot_y = np.array( (-1/theta[2])*(theta[1]*plot_x + -0.5*theta[0]) ) 

plt.plot( plot_x, plot_y, label = 'Regressor Linear', color='green', linestyle='-', linewidth=1.5) 


# <span style='font-family:serif'>
#     
# ## 7. Dados Não-Lineares e Regularização
#     
# Na maioria dos casos da vida real, os dados não são linearmente separáveis. 

# In[207]:


df = pd.read_csv( 'dados2.csv', sep=',')


# In[208]:


df.head()


# In[209]:


X = df.iloc[:, 0:-1].values

Y = df.iloc[:, -1].values #TARGET

print('X:', X[0:5,:])

print('Y:', Y[0:5])


# In[210]:


visualizaDados(X,Y)

plt.show()


# Para realizar a regressão logística neste caso, teremos que transformar os atributos originais em atributos polinomiais.

# In[213]:


def atributosPolinomiais(X1,X2):
    """
    Gera atributos polinomiais a partir dos atriburos
    originais da base. 
 
    Retorna um novo vetor de mais atributos:
    X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2
 
    As entradas X1, X2 devem ser do mesmo tamanho.
    """
    
    grau=15
    
    if not isinstance(X1,  np.ndarray):
        X1 = np.array( [[X1]] )
        X2 = np.array( [[X2]] )
        
    out = np.ones( len(X1) )
    
    for i in range( 1,grau+1 ):
        for j in range( 0,i+1 ):
            out_temp = ( X1**(i-j) ) * (X2**j)
            
            out = np.column_stack( (out,out_temp) ) 
            
    return out

X_poli = atributosPolinomiais(X[:,0],X[:,1]) 

print('Dimensão dos atributos do novo conjunto de dados polinomiais: \n', X_poli.shape[1])


# Quando aumentamos o grau dos atributos, torna-se necessário adicionar um parâmetro de regularização a função de custo para evitar o overfitting dos dados. A regularização adiciona à função custo o valor dos parâmetros. Tal adição resulta na eliminação de parâmetros de pouca importância e, portanto, em um modelo mais convexo, do qual que se espera que seja mais representativo da realidade. 
# 
# $$ J(\theta) = \left[ \frac{1}{m} \sum_{i=1}^{m} \left[-y^{(i)} \log\left(h_\theta(x^{(i)})\right) - \left(1 - y^{(i)}\right) \log\left(1 - h_\theta(x^{(i)})\right) \right] \right] + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_{j}^2. $$
# 
# O gradiente da função de custo é um vetor no qual o $j-$ésimo elemento é definido como:
# 
# $$\frac{\partial J(\theta)}{\partial \theta_j}= 
# \begin{cases}
#     \displaystyle \frac{1}{m} \sum_{i=1}^{m} \left(h_\theta(x^{(i)}) - y^{(i)} \right).x_{j}^{(i)},& \text{se } j = 0\\
#     \displaystyle \left(\frac{1}{m} \sum_{i=1}^{m} \left(h_\theta(x^{(i)}) - y^{(i)} \right).x_{j}^{(i)}\right) + \frac{\lambda}{m} \theta_j,              & \text{se } j \geq 0\\
# \end{cases}
# $$

# In[214]:


def funcaoCustoRegEGrad(theta, X, Y, lambda_reg):

    """
    Calcula o custo de usar theta como parametro da regressao logistica 
    para ajustar os dados de X e Y. Computa tambem as derivadas parciais 
    para o custo com relacao ao parametro theta. 
    """
    
    m = len(Y) #numero de exemplos de treinamento

    J = 0
    grad = np.zeros( len(theta) )
    
    eps = 1e-15
    
    theta2 = theta[1:]
    
    J = (1 / m) * np.sum(-Y * np.log(eps+sigmoid(np.dot(X, theta))) - (1 - Y) * np.log(1 - sigmoid(np.dot(X, theta)) + eps))+((lambda_reg/(2*m))*np.sum(theta2*theta2))   
    
    for t in range(len(theta)):
        
        if t==0:
            
            grad[t] = (1/m) * np.dot(X[:,t].T, sigmoid(np.dot(X, theta)) - Y)
    
        else:
            grad[t] = ((1/m) * np.dot(X[:,t].T, sigmoid(np.dot(X, theta)) - Y))+(lambda_reg/m)*theta[t]
    
    
    return J, grad



m, n = X_poli.shape

theta = np.zeros(n)

lambda_reg = 1

custo, grad = funcaoCustoRegEGrad(theta, X_poli, Y, lambda_reg)

print('Custo encontrado para theta inicial (zeros): ',custo)

print('Vetor de Gradiente: ',grad)


# <span style='font-family:serif'>
#     
# ## 8. Regularização Fraca (0.00005)
# 
# 

# In[215]:



lambda_reg = 0.00005

iteracoes = 50
theta = np.zeros(n) 

result = scipy.optimize.minimize(fun=funcaoCustoRegEGrad, x0=theta, args=(X_poli, Y, lambda_reg),  
                method='Newton-CG',jac=True, options={'maxiter': iteracoes, 'disp':True})

theta0 = result.x

custo, grad = funcaoCustoRegEGrad(theta, X_poli, Y, lambda_reg) 


# In[216]:


visualizaDados(X,Y)

u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)

z = np.zeros( [len(u), len(v)] )

for i in range( len(u) ):
    for j in range( len(v) ):
        z[i,j] = np.dot( atributosPolinomiais( u[i], v[j] ),theta0);

plt.contour(u, v, z, levels=[0], cmap=plt.cm.Paired)


plt.show()


# <span style='font-family:serif'>
#     
# ## 9. Regularização Forte (5)
# 
# 

# In[221]:


lambda_reg = 5

iteracoes = 50
theta = np.zeros(n) 

result = scipy.optimize.minimize(fun=funcaoCustoRegEGrad, x0=theta, args=(X_poli, Y, lambda_reg),  
                method='Newton-CG',jac=True, options={'maxiter': iteracoes, 'disp':True})

theta = result.x

custo, grad = funcaoCustoRegEGrad(theta, X_poli, Y, lambda_reg) 

visualizaDados(X,Y)

u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)

z = np.zeros( [len(u), len(v)] )

for i in range( len(u) ):
    for j in range( len(v) ):
        z[i,j] = np.dot( atributosPolinomiais( u[i], v[j] ),theta);

plt.contour(u, v, z, levels=[0], cmap=plt.cm.Paired)


plt.show()

