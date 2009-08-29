// RUN: clang-cc -fsyntax-only -verify %s

template<typename T>
struct A {
  void f();
};

template<typename T>
struct B : A<T> {
  using A<T>::f;
  
  void g() {
    f();
  }
};

template struct B<int>;
