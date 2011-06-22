// RUN: %clang_cc1 -fsyntax-only -verify %s
template <typename T> struct S {
  S() { }
  S(T t);
};

template struct S<int>;

void f() {
  S<int> s1;
  S<int> s2(10);
}

namespace PR7184 {
  template<typename T>
  void f() {
    typedef T type;
    void g(int array[sizeof(type)]);
  }

  template void f<int>();
}

namespace UsedAttr {
  template<typename T>
  void __attribute__((used)) foo() {
    T *x = 1; // expected-error{{cannot initialize a variable of type 'int *' with an rvalue of type 'int'}}
  }

  void bar() {
    foo<int>(); // expected-note{{instantiation of}}
  }
}

namespace PR9654 {
  typedef void ftype(int);

  template<typename T>
  ftype f;

  void g() {
    f<int>(0);
  }
}
