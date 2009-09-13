// RUN: clang-cc -fsyntax-only -verify %s
template<typename T> struct A {
  struct B { };
  
  friend struct B;
};

void f() {
  A<int>::B b;
}
