// RUN: clang -E %s | grep '^a: x$' &&
// RUN: clang -E %s | grep '^b: x y, z,h$'

#define A(b, c...) b c
a: A(x)
b: A(x, y, z,h)

