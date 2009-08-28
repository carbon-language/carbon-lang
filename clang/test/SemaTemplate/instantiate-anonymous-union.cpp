// RUN: clang-cc -fsyntax-only %s

// FIXME: We need to test anonymous structs/unions in templates for real.

template <typename T> class A { struct { }; };

A<int> a0;

