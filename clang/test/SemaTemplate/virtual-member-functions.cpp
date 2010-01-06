// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace PR5557 {
template <class T> struct A {
  A();
  virtual void anchor(); // expected-note{{instantiation}}
  virtual int a(T x);
};
template<class T> A<T>::A() {}
template<class T> void A<T>::anchor() { }

template<class T> int A<T>::a(T x) { 
  return *x; // expected-error{{requires pointer operand}}
}

void f(A<int> x) {
  x.anchor();
}

template<typename T>
struct X {
  virtual void f();
};

template<>
void X<int>::f() { }
}

template<typename T>
struct Base {
  virtual ~Base() { 
    int *ptr = 0;
    T t = ptr; // expected-error{{cannot initialize}}
  }
};

template<typename T>
struct Derived : Base<T> {
  virtual void foo() { }
};

template struct Derived<int>; // expected-note{{instantiation}}

