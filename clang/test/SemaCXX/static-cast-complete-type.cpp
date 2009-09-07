// RUN: clang-cc -fsyntax-only -verify %s
template<typename T> struct S {
  S(int);
};

struct T; // expected-note{{forward declaration of 'struct T'}}

void f() {
  S<int> s0 = static_cast<S<int> >(0);
  S<void*> s1 = static_cast<S<void*> >(00);

  (void)static_cast<T>(10); // expected-error{{'struct T' is an incomplete type}}
}
