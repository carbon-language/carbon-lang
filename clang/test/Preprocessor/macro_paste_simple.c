// RUN: %clang_cc1 %s -E | grep "barbaz123"

#define FOO bar ## baz ## 123

FOO
