// RUN: clang-cc %s -E | grep 'pre: 1 1 X' &&
// RUN: clang-cc %s -E | grep 'nopre: 1A(X)'

/* Preexpansion of argument. */
#define A(X) 1 X
pre: A(A(X))

/* The ## operator disables preexpansion. */
#undef A
#define A(X) 1 ## X
nopre: A(A(X))

