# ndarray-householder-qr-complex

[![Build Status](https://travis-ci.org/scijs/ndarray-householder-qr-complex.svg?branch=master)](https://travis-ci.org/scijs/ndarray-householder-qr-complex) [![npm version](https://badge.fury.io/js/ndarray-householder-qr-complex.svg)](http://badge.fury.io/js/ndarray-householder-qr-complex) [![Dependency Status](https://david-dm.org/scijs/ndarray-householder-qr-complex.svg)](https://david-dm.org/scijs/ndarray-householder-qr-complex)

A module for calculating the in-place [QR decomposition](http://en.wikipedia.org/wiki/QR_decomposition) of an ndarray using Householder triangularization

## Introduction

The algorithm is the Householder QR Factorization algorithm as found on p. 73 of Trefethen and Bau's [Numerical Linear Algebra](http://www.amazon.com/Numerical-Linear-Algebra-Lloyd-Trefethen/dp/0898713617). In pseudocode, the algorithm is:

```
for k = 1 to n
  x = A[k:m,k]
  v_k = sign(x_1) ||x||_2 e_1 + x
  v_k = v_k / ||v_k||_2
  A[k:m,k:n] = A[k:m,k:n] - 2 v_k (v_k^* A[k:m,k:n])
```

The specific implementation is based on the pseudocode from Walter Gander's [Algorithms for the QR-Decomposition](http://www.inf.ethz.ch/personal/gander/papers/qrneu.pdf). This algorithm computes both R and the Householder reflectors in place, storing R in the upper-triangular portion of A, the diagonal of R in a separate vector and the Householder reflectors in the columns of A. To eliminate unnecessary operations, the Householder reflectors are normalized so that norm(v) = sqrt(2).

## Example

For an example, see [ndarray-householder-qr](https://github.com/scijs/ndarray-householder-qr). The only difference for the complex extension is that all arrays are doubled up with real and imaginary components in separate arrays. For some hints on working with complex numbers in ndarrays, see [ndarray-blas-level1-complex](https://github.com/scijs/ndarray-blas-level1-complex).


## Usage

##### `factor( A_r, A_i, d_r, d_i )`
Computes the in-place triangularization of `A` given `A_r` and `A_i` (the real and imaginary components of `A`), returning the Householder reflectors in the lower-triangular portion of `A` (including the diagonal) and `R` in the upper-triangular portion of `A` (excluding diagonal) with the diagonal of `R` stored in `d_r` and `d_i`. `d_r` and `d_i` must be one-dimensional vectors with length at least `n`.

##### `multByQ( A_r, A_i, x_r, x_i )`
Compute the product Q * x in-place, replacing x with Q * x. `A_r` and `A_i` are the real/imag components of the in-place factored matrix.

##### `multByQinv( A_r, A_i, x_r, x_i )`
`A` is the in-place factored matrix. Compute the product `Q^-1 * x` in-place, replacing x with `Q^-1 * x`. Since the product is shorter than `x` for m > n, the entries of `x` from n+1 to m will be zero.

##### `constructQ( A_r, A_i, Q_r, Q_i )`
Given the in-place factored matrix A (diagonal not necessary), construct the matrix Q by applying the reflectors to a sequence of unit vectors. The dimensions of Q must be between m x n and m x m. When the dimensions of Q are m x n, Q corresponds to the Reduced QR Factorization. When the dimensions are m x m, Q corresponds to the Full QR Factorization.

##### `factor( A_r, A_i, Q_r, Q_i )`
**Incomplete**
Compute the in-place QR factorization of A, storing R in A and outputting Q in Q.

##### `solve( A_r, A_i, d_r, d_i, x_r, x_i )`
Use the previously-calculated triangularization to find the vector x that minimizes the L-2 norm of (Ax - b). Note that the vector b is modified in the process.
- `A_r` and `A_i` are the real/imag components of the in-place factored matrix computed by `factor`
- `d_r` and `d_i` are the real/imag components of the diagonal of `R` computed by `factor`
- `x_r` and `x_i` are the real/imag components of the input vector of length m. The answer is computed in-place in the first n entries of `x`. The remaining entries are zero.


## Credits
(c) 2015 Ricky Reusser. MIT License
