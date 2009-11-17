// RUN: clang-cc -fsyntax-only -verify %s

namespace N { }

template<typename T>
struct A {
  void f();
};

template<typename T>
struct B : A<T> {
  using A<T>::f;
  
  void g() {
    using namespace N;
    f();
  }
};

template struct B<int>;
