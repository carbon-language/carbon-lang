<!--
Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
-->

This note attempts to describe the motivation for and design of an
implementation of Fortran 90 (and later) array expression evaluation that
minimizes the use of dynamically allocated temporary storage for
the results of calls to transformational intrinsic functions.

The transformational intrinsic functions of Fortran of interest to
us here include:

* Reductions to scalars (`SUM(X)`, also `ALL`, `ANY`, `COUNT`, `IALL`,
  `IANY`, `IPARITY`, `MAXVAL`, `MINVAL`, `PARITY`, `PRODUCT`)
* Axial reductions (`SUM(X,DIM=)`, &c.)
* Location reductions to indices (`MAXLOC`, `MINLOC`, `FINDLOC`)
* Axial location reductions (`MAXLOC(DIM=`, &c.)
* `TRANSPOSE(M)` matrix transposition
* `RESHAPE` without `ORDER=`
* `RESHAPE` with `ORDER=`
* `CSHIFT` and `EOSHIFT` with scalar `SHIFT=`
* `CSHIFT` and `EOSHIFT` with array-valued `SHIFT=`
* `PACK` and `UNPACK`

Other Fortran intrinsic functions are technically transformational (e.g.,
`COMMAND_ARGUMENT_COUNT`) but not of interest for this note.
The generic `REDUCE` is also not considered here.

Arrays as functions
===================
A whole array can be viewed as a function that maps its indices to the values
of its elements.
Specifically, it is a map from a tuple of integers to its element type.
The rank of the array is the number of elements in that tuple,
and the shape of the array delimits the domain of the map.

`REAL :: A(N,M)` can be seen as a function mapping ordered pairs of integers
`(J,K)` with `1<=J<=N` and `1<=J<=M` to real values.

Array expressions as functions
==============================
The same perspective can be taken of an array expression comprising
intrinsic operators and elemental functions.
Fortran doesn't allow one to apply subscripts directly to an expression,
but expressions have rank and shape, and one can view array expressions
as functions over index tuples by applying those indices to the arrays
in the expression.

Consider `B = A + 1.0` (assuming `REAL :: A(N,M), B(N,M)`).
The right-hand side of that assignment could be evaluated into a
temporary array `T` and then subscripted as it is copied into `A`.
```
REAL, ALLOCATABLE :: T(:,:)
ALLOCATE(T(N,M))
FORALL(J=1:N,K=1:M) T(J,K)=A(J,K) + 1.0
FORALL(J=1:N,K=1:M) B(J,K)=T(J,K)
DEALLOCATE(T(N,M))
```
But we can avoid the allocation, population, and deallocation of
the temporary by treating the right-hand side expression as if it
were a statement function `F(J,K)=A(J,K)+1.0` and evaluating
```
FORALL(J=1:N,K=1:M) A(J,K)=F(J,K)
```

In general, when a Fortran array assignment to a non-allocatable array
does not include the left-hand
side variable as an operand of the right-hand side expression, and any
function calls on the right-hand side are elemental or scalar-valued,
we can avoid the use of a temporary.

Transformational intrinsic functions as function composition
============================================================
Many of the transformational intrinsic functions listed above
can, when their array arguments are viewed as functions over their
index tuples, be seen as compositions of those functions with
functions of the "incoming" indices.

For example, the application of `TRANSPOSE(A + 1.0)` to the index
tuple `(J,K)` becomes `A(K,J) + 1.0`.

Determination of rank and shape
===============================
An important part of evaluating array expressions without the use of
temporary storage is determining the shape of the result prior to,
or without, evaluating the elements of the result.

The shapes of array objects, results of elemental intrinsic functions,
and results of intrinsic operations are obvious.
But it is possible to determine the shapes of the results of many
transformantional intrinsic function calls as well.

* `SHAPE(SUM(X,DIM=d))` is `SHAPE(X)` with one element removed.
  The `DIM=` argument is commonly a compile-time constant.
* `SHAPE(MAXLOC(X))` is `[RANK(X)]`.
* `SHAPE(MAXLOC(X,DIM=d))` is `SHAPE(X)` with one element removed.
* `SHAPE(TRANSPOSE(M))` is a reversal of `SHAPE(M)`.
* `SHAPE(RESHAPE(..., SHAPE=S))` is `S`.
* `SHAPE(CSHIFT(X))` is `SHAPE(X)`; same with `EOSHIFT`.
* `SHAPE(PACK(A,VECTOR=V))` is `SHAPE(V)`; `RANK(PACK(...))` is always 1.
* `SHAPE(UNPACK(MASK=M))` is `SHAPE(M)`.
* `SHAPE(SHAPE(X))` is `[RANK(X)]`.

This is useful because expression evaluations that *do* require temporaries
to hold their results (due to the context in which the evaluation occurs)
can be implemented with a separation of the allocation
of the temporary array and the population of the array.
The code that evaluates the expression, or that implements a transformational
intrinsic in the runtime library, can be designed with an API that includes
a pointer to the destination array as an argument.

Statements like `ALLOCATE(A,SOURCE=expression)` should thus be capable
of evaluating their array expressions directly into the newly-allocated
storage for the allocatable array.
The implementation would generate code to calculate the shape, use it
to allocate the memory and populate the descriptor, and then drive a
loop nest around the expression to populate the array.
In cases where the analyzed shape is known at compile time, we should
be able to have the opportunity to avoid heap allocation in favor of
stack storage, if the scope of the variable is local.

Automatic reallocation of allocatables
======================================
Fortran 2003 introduced the ability to assign non-conforming array expressions
to ALLOCATABLE arrays with the implied semantics of reallocation to the
new shape.
The implementation of this feature also becomes more straightforward if
our implementation of array expressions has decoupled calculation of shapes
from the evaluation of the elements of the result.

