# Reconstructing noisy image by dictionary learning
Our goal is to use tools of linear algebra (MP,OMP,MOD) to reconstruct the noisy image and retrieve the original one. 


## What is dictionary learning?
Dictionary learning is the method of learning a matrix, called a dictionary, such that we can write a signal as a linear combination of as few columns from the matrix as possible.

When using dictionary learning for images we take advantage of the property that natural images can be represented in a sparse way. This means that if we have a set of basic image features any image can be written as a linear combination of only a few basic features. The matrix we call a dictionary is such a set. Each column of the dictionary is one basic image features. In the litterature these feature vectors are called atoms.

With dictionary learning we want to find a dictionary, $D$, and a vector with coefficients for the linear combinations, $x$. The vector of an image patches we’ll denote by $y$. Then our goal is to find $D$ and $x$ such that the $error$ $∥y−Dx∥_{2}$ is small. 

## How to update sparse vectors $x$ , assuming that we have a perfect ditionary?

for each $x_{i}$ we have following least square problem : 

$$x_{i}^{k}=argmin_{x_{i}}||y_{i}-Dx_{i}^{k-1}||^{2} \ \ \ \ s.t   \ \ \ ||x||_{0}\le k $$

we know that above problem can be solved using MP(Matching Pursuit). we also can use OMP(Orthogonal Matching Pursuit) if we want better answer. 

## what is MP?

Matching pursuit (MP) is a sparse approximation algorithm which finds the "best matching" projections of multidimensional data onto the span of an over-complete dictionary. you can see 
