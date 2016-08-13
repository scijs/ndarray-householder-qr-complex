'use strict'

var cblas = require('ndarray-blas-level1-complex'),
    sblas = require('ndarray-blas-level1'),
    ops = require('ndarray-ops'),
    diag = require('ndarray-diagonal'),
    cmod = require('complex-modulus'),
    cdiv = require('complex-division')

function show(label,data) {
  if( typeof data === 'number' ) {
    console.log(label+' = '+data)
  } else if( data.dimension === 1 ) {
    console.log(label+' = '+ndshow(data))
  } else {
    console.log(label+' =\n'+ndshow(data.transpose(1,0))+'\n')
  }
}


function factor ( A_r, A_i, d_r, d_i ) {
  var j,m,n,i, s, z, ajj_r, ajj_i, ajj_nrm, Ajmj_r, Ajmj_i, dj_r, dj_i,
      Ajmi_r, Ajmi_i, fak, d_rOff, d_iOff, Ajmi_rOff, Ajmi_iOff,
      Ajmi_rInc, Ajmi_iInc, Ajmj_rInc, Ajmj_iInc, Ajmi_rInc1, Ajmi_iInc1,
      Ajmi_rInc2, Ajmi_iInc2, Ajmj_rShape, Ajmj_iShape, d_rInc, d_iInc

  if( A_r.dimension !== 2 ) {
    throw new TypeError('factor():: dimension of input matrix must be 2.')
  }

  m = A_r.shape[0]
  n = A_r.shape[1]

  if( m < n ) {
    throw new TypeError('factor():: In input matrix A, number of rows m must be greater than number of column n.')
  }

  Ajmj_r = A_r.pick(null,0)
  Ajmj_i = A_i.pick(null,0)

  Ajmi_r = A_r.pick(null,0)
  Ajmi_i = A_i.pick(null,0)

  Ajmi_rOff = Ajmi_r.offset
  Ajmi_iOff = Ajmi_i.offset

  Ajmj_rInc = A_r.stride[0] + A_r.stride[1]
  Ajmj_iInc = A_i.stride[0] + A_i.stride[1]

  Ajmi_rInc1 = A_r.stride[0]
  Ajmi_iInc1 = A_i.stride[0]

  Ajmi_rInc2 = A_r.stride[1]
  Ajmi_iInc2 = A_i.stride[1]

  Ajmj_rShape = Ajmj_r.shape
  Ajmj_iShape = Ajmj_i.shape

  d_rInc = d_r.stride[0]
  d_iInc = d_i.stride[0]

  d_rOff = d_r.offset
  d_iOff = d_i.offset

  for( j=0;
       j<n;
       j++,
         Ajmj_rShape[0]--,
         Ajmj_iShape[0]--,
         Ajmj_r.offset+=Ajmj_rInc,
         Ajmj_i.offset+=Ajmj_iInc,
         Ajmi_rOff+=Ajmi_rInc1,
         Ajmi_iOff+=Ajmi_iInc1,
         d_rOff+=d_rInc,
         d_iOff+=d_iInc
    ) {

    s = cblas.nrm2(Ajmj_r, Ajmj_i)

    ajj_r = Ajmj_r.data[Ajmj_r.offset]
    ajj_i = Ajmj_i.data[Ajmj_i.offset]

    ajj_nrm = cmod( ajj_r, ajj_i )

    dj_r = - s * ajj_r / ajj_nrm
    dj_i = - s * ajj_i / ajj_nrm

    d_r.data[d_rOff] = dj_r
    d_i.data[d_iOff] = dj_i

    s = Math.sqrt( s * (s + ajj_nrm) )

    if( s === 0 ) return false

    Ajmj_r.data[Ajmj_r.offset] = ajj_r - dj_r
    Ajmj_i.data[Ajmj_i.offset] = ajj_i - dj_i

    sblas.scal( 1/s, Ajmj_r)
    sblas.scal( 1/s, Ajmj_i)

    for(i=j+1,
          Ajmi_r.offset=Ajmi_rOff + Ajmi_rInc2*i,
          Ajmi_i.offset=Ajmi_iOff + Ajmi_iInc2*i;
        i<n;
        i++,
          Ajmi_r.offset+=Ajmi_rInc2,
          Ajmi_i.offset+=Ajmi_iInc2
    ) {
      z = cblas.doth( Ajmj_r, Ajmj_i, Ajmi_r, Ajmi_i )
      cblas.axpy( -z[0], -z[1], Ajmj_r, Ajmj_i, Ajmi_r, Ajmi_i )
    }

    Ajmi_r.shape[0]--;
    Ajmi_i.shape[0]--;
  }

  return true
}


var multiplyByQ = function multiplyByQ ( A_r, A_i, b_r, b_i ) {
  var j, z, Ajmj_r, Ajmj_i, yk_r, yk_i, n=A_r.shape[1], m=A_i.shape[0],
      Ajmj_rOffInc, Ajmj_iOffInc, Ajmj_rShape, Ajmj_iShape,
      yk_rInc, yk_iInc, yk_rShape, yk_iShape

  Ajmj_r = A_r.pick(null,n-1).lo(n-1)
  Ajmj_i = A_i.pick(null,n-1).lo(n-1)
  yk_r = b_r.lo(n-1)
  yk_i = b_i.lo(n-1)

  Ajmj_rOffInc = A_r.stride[0] + A_r.stride[1]
  Ajmj_iOffInc = A_i.stride[0] + A_i.stride[1]
  Ajmj_rShape = Ajmj_r.shape
  Ajmj_iShape = Ajmj_i.shape
  yk_rInc = yk_r.stride[0]
  yk_iInc = yk_i.stride[0]
  yk_rShape = yk_r.shape
  yk_iShape = yk_i.shape

  for(j=n-1;
      j>=0;
      j--,
        Ajmj_r.offset-=Ajmj_rOffInc,
        Ajmj_i.offset-=Ajmj_iOffInc,
        Ajmj_rShape[0]++,
        Ajmj_iShape[0]++,
        yk_r.offset-=yk_rInc,
        yk_i.offset-=yk_iInc,
        yk_rShape[0]++,
        yk_iShape[0]++
  ) {
    z = cblas.doth(Ajmj_r, Ajmj_i, yk_r, yk_i)
    cblas.axpy( -z[0], -z[1], Ajmj_r, Ajmj_i, yk_r, yk_i)
  }

  return true
}

var multiplyByQinv = function multiplyByQinv( A_r, A_i, b_r, b_i ) {
  var j, z, Ajmj_r, Ajmj_i, yk_r, yk_i, n=A_r.shape[1], m=A_r.shape[0],
    Ajmj_rOffInc, Ajmj_iOffInc, Ajmj_rShape, Ajmj_iShape,
    yk_rInc, yk_iInc, yk_rShape, yk_iShape

  Ajmj_r = A_r.pick(null,0).lo(0)
  Ajmj_i = A_i.pick(null,0).lo(0)
  yk_r = b_r.lo(0)
  yk_i = b_i.lo(0)

  Ajmj_rOffInc = A_r.stride[0] + A_r.stride[1]
  Ajmj_iOffInc = A_i.stride[0] + A_i.stride[1]
  Ajmj_rShape = Ajmj_r.shape
  Ajmj_iShape = Ajmj_i.shape
  yk_rInc = yk_r.stride[0]
  yk_iInc = yk_i.stride[0]
  yk_rShape = yk_r.shape
  yk_iShape = yk_i.shape

  for(j=0;
      j<n;
      j++,
        Ajmj_r.offset+=Ajmj_rOffInc,
        Ajmj_i.offset+=Ajmj_iOffInc,
        Ajmj_rShape[0]--,
        Ajmj_iShape[0]--,
        yk_r.offset+=yk_rInc,
        yk_i.offset+=yk_iInc,
        yk_rShape[0]--,
        yk_iShape[0]--
  ) {
    z = cblas.doth(Ajmj_r, Ajmj_i, yk_r, yk_i)
    cblas.axpy( -z[0], -z[1], Ajmj_r, Ajmj_i, yk_r, yk_i)
  }

  return true
}

var constructQ = function( QR_r, QR_i, Q_r, Q_i ) {
  var j, Qj_r, Qj_i, n=Q_r.shape[1], m=Q_r.shape[0], Qj_rInc, Qj_iInc


  ops.assigns(Q_r,0)
  ops.assigns(Q_i,0)
  ops.assigns(diag(Q_r),1)

  Qj_r = Q_r.pick(null,0)
  Qj_i = Q_i.pick(null,0)
  Qj_rInc = Q_r.stride[1]
  Qj_iInc = Q_i.stride[1]

  for(j=0; j<n; j++, Qj_r.offset+=Qj_rInc, Qj_i.offset+=Qj_iInc ) {
    multiplyByQ( QR_r, QR_i, Qj_r, Qj_i )
  }

  return true
}

//var factorize = function( A, Q, R ) {
//}


var solve = function( QR_r, QR_i, d_r, d_i, x_r, x_i ) {
  var m,n,j,QRi, QRi, QRiInc, QRiShape, dotu=cblas.dotu, xj, xjInc, xjShape, xInc,
      xPos, dPos, dInc, xData, dData, dPos, dInc, z = new Array(2), d, tmp_i, tmp_r,
      d_iPos, QR_ri, QR_ii, QR_riInc, QR_iiInc, QR_riShape, QR_iiShape, xj_r, xj_i,
      xj_rInc, xj_iInc, xj_rShape, xj_iShape, x_rPos, x_iPos, x_rInc, x_iInc,
      x_rData, x_iData, d_rPos, d_iPos, d_rInc, d_iInc, d_rData, d_iData


  if( QR_r.dimension !== 2 ) {
    throw new TypeError('factor():: dimension of input matrix must be 2.')
  }

  m = QR_r.shape[0]
  n = QR_r.shape[1]

  multiplyByQinv( QR_r, QR_i, x_r, x_i )

  QR_ri = QR_r.pick(n-2,null).lo(n-1)
  QR_ii = QR_i.pick(n-2,null).lo(n-1)
  QR_riInc = QR_r.stride[1] + QR_r.stride[0]
  QR_iiInc = QR_i.stride[1] + QR_i.stride[0]
  QR_riShape = QR_ri.shape
  QR_iiShape = QR_ii.shape

  xj_r = x_r.lo(n-1)
  xj_i = x_i.lo(n-1)
  xj_rInc = x_r.stride[0]
  xj_iInc = x_i.stride[0]
  xj_rShape = xj_r.shape
  xj_iShape = xj_i.shape

  x_rPos = x_r.offset + x_r.stride[0]*(n-1)
  x_iPos = x_i.offset + x_i.stride[0]*(n-1)
  x_rInc = x_r.stride[0]
  x_iInc = x_i.stride[0]
  x_rData = x_r.data
  x_iData = x_i.data

  d_rPos = d_r.offset + d_r.stride[0]*(n-1)
  d_iPos = d_i.offset + d_i.stride[0]*(n-1)
  d_rInc = d_r.stride[0]
  d_iInc = d_i.stride[0]
  d_rData = d_r.data
  d_iData = d_i.data

  cdiv( x_rData[x_rPos], x_iData[x_iPos], d_rData[d_rPos], d_iData[d_iPos], z )
  x_rData[x_rPos] = z[0]
  x_iData[x_iPos] = z[1]

  //xData[xPos] = xData[xPos] / dData[dPos]

  for(j=n-2,
        x_rPos-=x_rInc,
        x_iPos-=x_iInc,
        d_rPos -= d_rInc,
        d_iPos -= d_iInc;
      j>=0;
      j--,
        QR_ri.offset-=QR_riInc,
        QR_ii.offset-=QR_iiInc,
        QR_riShape[0]++,
        QR_iiShape[0]++,
        xj_rShape[0]++,
        xj_iShape[0]++,
        xj_r.offset-=xj_rInc,
        xj_i.offset-=xj_iInc,
        x_rPos-=x_rInc,
        x_iPos-=x_iInc,
        d_rPos-=d_rInc,
        d_iPos-=d_iInc
    ) {

    d = dotu( QR_ri, QR_ii, xj_r, xj_i )
    tmp_r = x_rData[x_rPos] - d[0]
    tmp_i = x_iData[x_iPos] - d[1]
    cdiv(tmp_r, tmp_i, d_rData[d_rPos], d_iData[d_iPos], z)
    x_rData[x_rPos] = z[0]
    x_iData[x_iPos] = z[1]
  }

  return true
}


exports.factor = factor
exports.multiplyByQ = multiplyByQ
exports.multiplyByQinv = multiplyByQinv
exports.constructQ = constructQ
//exports.factorize = factorize
exports.solve = solve


// Deprecations:
exports.triangularize = function() {
  console.warn('Warning: ndarray-householder-qr-complex::triangularize() has been deprecated and renamed factor().')
  return exports.factor.apply(this,arguments)
}
