// RUN: %clang_cc1 -verify %s

@interface A
@end

template<typename T>
struct X0 {
  void f(T); // expected-error{{interface type 'A' cannot be passed by value}}
};

X0<A> x0a; // expected-note{{instantiation}}

