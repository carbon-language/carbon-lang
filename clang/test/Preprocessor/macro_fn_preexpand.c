// clang %s -E | grep '1 1 X'
/* Preexpansion of argument.*/
#define A(X) 1 X
A(A(X))

