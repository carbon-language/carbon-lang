// RUN: clang-cc -fsyntax-only %s

template <class A> int x(A x) { return x++; }
int y() { return x<int>(1); }
