// RUN: clang-cc -fsyntax-only -verify %s

extern "C" { void f(bool); }

namespace std {
  using ::f;
  inline void f() { return f(true); }
}

namespace M {
  void f(float);
}

namespace N {
  using M::f;
  void f(int) { } // expected-note{{previous}}
  
  void f(int) { } // expected-error{{redefinition}}
}
