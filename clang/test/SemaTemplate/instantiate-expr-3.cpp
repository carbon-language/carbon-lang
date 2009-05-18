// RUN: clang-cc -fsyntax-only -verify %s

template<typename T>
struct ImaginaryLiteral0 {
  void f(T &x) {
    x = 3.0I; // expected-error{{incompatible type}}
  }
};

template struct ImaginaryLiteral0<_Complex float>;
template struct ImaginaryLiteral0<int*>; // expected-note{{instantiation}}
