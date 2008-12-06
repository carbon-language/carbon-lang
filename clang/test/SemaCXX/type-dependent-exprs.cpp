// RUN: clang -fsyntax-only -verify %s

void g(int);

template<typename T>
T f(T x) {
  (void)(x + 0);
  (void)T(0);
  (void)(x += 0);
  (void)(x? x : x);
  return g(x);
  h(x); // h is a dependent name
  g(1, 1); // expected-error{{too many arguments to function call}}
  h(1); // expected-error{{use of undeclared identifier 'h'}}
  return 0;
}
