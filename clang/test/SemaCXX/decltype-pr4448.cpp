// RUN: clang-cc -fsyntax-only -verify %s -std=c++0x

template< typename T, T t, decltype(t+2) v >
struct Convoluted {};

int test_array[5];

Convoluted< int *, test_array, nullptr > tarray;
