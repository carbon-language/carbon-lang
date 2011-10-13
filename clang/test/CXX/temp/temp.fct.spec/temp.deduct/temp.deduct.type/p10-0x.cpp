// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
template<typename T> void f(T&&);
template<> void f(int&) { }
void (*fp)(int&) = &f;
