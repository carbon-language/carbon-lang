// RUN: %clang_cc1 -fsyntax-only -verify %s
class X { 
public:
  virtual int f();
};

void g(int); // expected-note{{candidate function}}

template<typename T>
T f(T x) {
  (void)(x + 0);
  (void)T(0);
  (void)(x += 0);
  (void)(x? x : x);
  (void)static_cast<int>(x);
  (void)reinterpret_cast<int>(x);
  (void)dynamic_cast<X*>(&x);
  (void)const_cast<int>(x);
  return g(x);
  h(x); // h is a dependent name
  g(1, 1); // expected-error{{no matching function for call}}
  h(1); // expected-error{{no matching function for call to 'h'}}
  return 0;
}
