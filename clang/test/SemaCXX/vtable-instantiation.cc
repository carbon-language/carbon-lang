// RUN: %clang_cc1 -fsyntax-only -verify %s
// PR8640

template<class T1> struct C1 {
  virtual void c1() {
    T1 t1 = 3;  // expected-error {{cannot initialize a variable}}
  }
};

template<class T2> struct C2 {
  void c2() {
    new C1<T2>();  // expected-note {{in instantiation of member function}}
  }
};

void f() {
  C2<int*> c2;
  c2.c2();  // expected-note {{in instantiation of member function}}
}

