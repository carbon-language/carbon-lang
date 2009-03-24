// RUN: clang-cc %s -E | grep 'foo(A, )'

#define X(Y) foo(A, Y)
X()

