// RUN: clang-cc -fsyntax-only %s

template<typename T> void f(T);

template<> void f(int) { }
void f(int) { }
