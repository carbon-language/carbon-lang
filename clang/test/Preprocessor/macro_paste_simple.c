// RUN: clang %s -E | grep "barbaz123"

#define FOO bar ## baz ## 123

FOO
