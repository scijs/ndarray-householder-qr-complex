'use strict'

var householder = require('../lib'),
    assert = require('chai').assert,
    ndarray = require('ndarray'),
    gemm = require("ndgemm"),
    ndshow = require('ndarray-show'),
    ndt = require('ndarray-tests'),
    fill = require('ndarray-fill'),
    pool = require('ndarray-scratch'),
    vander = require('ndarray-vandermonde'),
    blas = require('ndarray-blas-level1-complex'),
    diag = require('ndarray-diagonal'),
    cgemm = require('ndarray-blas-gemm-complex')

function show(label,data) {
  if( typeof data === 'number' ) {
    console.log(label+' = '+data)
  } else if( data.dimension === 1 ) {
    console.log(label+' = '+ndshow(data))
  } else {
    console.log(label+' =\n'+ndshow(data.transpose(1,0))+'\n')
  }
}


describe("Householder QR", function() {

  var m,n,A_r,A_i,b_r,b_i,x,d_r,d_i,Q_r,Q_i,R_r,R_i,QR_i,QR_r,A0_r,A0_i,b_r,b_i,x_r,x_i

  beforeEach(function() {
    m=3
    n=2

    x = ndarray([1,2,3])

    A0_r = vander(x,2)
    A0_i = ndarray([2,-4,3,2,-1,0],[3,2])
    A_r = vander(x,2)
    A_i = ndarray([2,-4,3,2,-1,0],[3,2])
    QR_r = pool.zeros([3,2])
    QR_i = pool.zeros([3,2])
    Q_r = pool.zeros([3,2])
    Q_i = pool.zeros([3,2])
    R_r = pool.zeros([2,2])
    R_i = pool.zeros([2,2])
    b_r = ndarray([1,2,3])
    b_r = ndarray([-2,-4,-3])
    d_r = pool.zeros([2])
    d_i = pool.zeros([2])

    x_r = ndarray([1,3,0])
    x_i = ndarray([-2,-8,0])

    b_r = ndarray([-24,29,8])
    b_i = ndarray([-20,-9,-27])
  })

  afterEach(function() {
    pool.free(d_r)
    pool.free(d_i)
  })

  it('decomposes and reconstructs the matrix',function() {
    assert( householder.triangularize(A_r,A_i,d_r,d_i), 'A is triangularized' )

    // Copy the upper-triangular part into a new matrix R:
    blas.copy( d_r, d_i, diag(R_r), diag(R_i) )
    for(var i=0;i<n;i++) {
      for(var j=i+1;j<n;j++) {
        R_r.set(i,j, A_r.get(i,j))
        R_i.set(i,j, A_i.get(i,j))
      }
    }

    assert( householder.constructQ( A_r, A_i, Q_r, Q_i ), 'Q is constructed' )

    cgemm(Q_r,Q_i,R_r,R_i,QR_r,QR_i)

    assert( ndt.approximatelyEqual( A0_r, QR_r, 1e-4 ), 'Re(Q*R) = Im(A)' )
    assert( ndt.approximatelyEqual( A0_i, QR_i, 1e-4 ), 'Im(Q*R) = Im(A)' )
  })

  it('solves a system of complex linear equations',function() {
    assert( householder.triangularize(A_r,A_i,d_r,d_i), 'A is triangularized' )
    assert( householder.solve(A_r,A_i,d_r,d_i,b_r,b_i), 'Solution succeeds' )
    assert( ndt.approximatelyEqual( b_r, x_r, 1e-4 ), 'Re(A^-1*b) = x' )
    assert( ndt.approximatelyEqual( b_i, x_i, 1e-4 ), 'Im(A^-1*b) = x' )
    

  })

})
