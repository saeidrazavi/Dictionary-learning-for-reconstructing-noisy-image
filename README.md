# Reconstructing noisy image by dictionary learning
Our goal is to use tools of linear algebra (MP,OMP,MOD) to reconstruct the noisy image and retrieve the original one. 

![1](https://user-images.githubusercontent.com/67091916/219641906-d430291c-8192-4e55-9ee6-a01c19c71e31.png)

## What is dictionary learning?
Dictionary learning is the method of learning a matrix, called a dictionary, such that we can write a signal as a linear combination of as few columns from the matrix as possible.

When using dictionary learning for images we take advantage of the property that natural images can be represented in a sparse way. This means that if we have a set of basic image features any image can be written as a linear combination of only a few basic features. The matrix we call a dictionary is such a set. Each column of the dictionary is one basic image features. In the litterature these feature vectors are called atoms.

With dictionary learning we want to find a dictionary, $D$, and a vector with coefficients for the linear combinations, $x$. The vector of an image patches we’ll denote by $y$. Then our goal is to find $D$ and $x$ such that the $error$ $∥y−Dx∥_{2}$ is small. 

## How to update sparse vectors $x$ , assuming that we have a perfect ditionary?

for each $x_{i}$ we have following least square problem : 

$$x_{i}^{k}=argmin_{x_{i}}||y_{i}-Dx_{i}^{k-1}||^{2} \ \ \ \ s.t   \ \ \ ||x||_{0}\le k $$

we know that above problem can be solved using MP(Matching Pursuit). we also can use OMP(Orthogonal Matching Pursuit) if we want better answer. 

## what is MP?

Matching pursuit (MP) is a sparse approximation algorithm which finds the "best matching" projections of multidimensional data onto the span of an over-complete dictionary. in each itteration, we map the residual, into space of each atom in dicionray, and choose atom that decreases the reconstruction error the most. we repeat this procedure until reconstruction error become less or equal to specific threshold. 

* implementation

```python
def MP(D: np.ndarray, Y: np.ndarray, tol:float) -> np.ndarray:
    
    threshold=D.shape[0] * tol ** 2
    num_sample=Y.shape[1]
    X=np.zeros([D.shape[1],num_sample])
    #--------------------------
    normilize_dic=normilize(D)  # it's good to normilize dictionary (helps us in time complexity)
    #--------------------------
    for i in range(num_sample):
        x_i=np.copy(X[:,i])
        residu=np.copy(Y[:,i]-np.dot(D,x_i))
        
        while(norm(residu)**2>threshold):
             #-------------------------
             error=norm(residu)**2-(np.dot(normilize_dic.T,residu))**2
             #---------------------------    
             indice=np.argmin(error)    
             #---------------------------
             z_star=np.dot(D[:,indice].T,residu)/(norm(D[:,indice])**2)
             x_i[indice]+=z_star
             residu=np.copy(Y[:,i]-np.dot(D,x_i))
        X[:,i]=x_i     
    
    return X
```

## what is OMP?
OMP is like MP with the differences that in each iteration, we make residual, independent of current atom that is used for reconstructing signal. 
this difference make this algorithm stronger that MP. although it's a little bit time consuming with respect to MP, but most of the time, we get better results.

* implementation
```python
def OMP(D: np.ndarray, Y: np.ndarray, tol:float) -> np.ndarray:

    threshold=D.shape[0] * tol ** 2
    num_sample=Y.shape[1]
    X=np.zeros([D.shape[1],num_sample])
    #--------------------------
    normilize_dic=normilize(D) # it's good to normilize dictionary (helps us in time complexity)
    #--------------------------
    for i in range(num_sample):
        indice_list=[]
        x_i=np.copy(X[:,i])
        residu=np.copy(Y[:,i]-np.dot(D,x_i))

        while(norm(residu)**2>threshold):
             #-------------------------
             error=norm(residu)**2-(np.dot(normilize_dic.T,residu))**2
             #---------------------------    
             indice=np.argmin(error)    
             #---------------------------
             indice_list.append(indice)
             D_k=D[:,indice_list]
             x_i=dot(pinv(D_k),Y[:,i])
             residu=np.copy(Y[:,i]-np.dot(D_k,x_i))
        #------------------------------      
        if(len(indice_list)!=0):
            X[indice_list,i]=x_i   
        else :
            X[:,i]=x_i      
    
    return X
```

# How to learn dictionary? 
suppose we have all $x_{i}'s$. now we want to learn dicitionary in a way that minimize reconstruction error. that is: 

$$D^{k}=agmin_{D} \ \sum_{i=1}^{N}||y_{i}-D^{k-1}x_{i}^{k}||$$

we can rewrite above error function as : 

$$D^{k}=agmin_{D} ||Y-D^{k-1}X^{k}||_{F}^{2}$$

$$where \ Y \ is \ concatinated \ form \ of \ y_{i}'s \ and \ X  \ is \ concatinated \ form \ of \ x_{i}'s$$

so we can find closed form solution for dicationary and update it in just one step : 

$$\boxed{D^{K}=YX^{(K)T}(X^{(K)}X^{(K)T})^{-1}}$$

$$where \ X^{(K)T}(X^{(K)}X^{(K)T})^{-1} \ is \ pseudo \ inverse \ of \ X$$

* implementation
```python
def MOD(Y:np.ndarray, X:np.ndarray):
    D=dot(dot(Y,X.T),pinv(dot(X,X.T)))
    return D
```

# MP results 

![2](https://user-images.githubusercontent.com/67091916/219654668-9985b5ed-03af-48f9-8754-7a4d6f3f04b5.png)
![3](https://user-images.githubusercontent.com/67091916/219654675-1b06b789-a0d9-4478-890a-09fdb0496744.png)
![4](https://user-images.githubusercontent.com/67091916/219654662-8f024633-a80f-410b-ae88-7e28526ed537.png)


## OMP results 

![2](https://user-images.githubusercontent.com/67091916/219654668-9985b5ed-03af-48f9-8754-7a4d6f3f04b5.png)
![5](https://user-images.githubusercontent.com/67091916/219655099-4a9d0260-5373-4ad6-9435-54afc95ba0c8.png)
![6](https://user-images.githubusercontent.com/67091916/219655091-f6d949fc-41c6-4c15-ba69-3b14c41553bf.png)
