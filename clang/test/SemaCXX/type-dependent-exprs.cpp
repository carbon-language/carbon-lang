// RUN: clang -fsyntax-only -verify %s

void g(int);

template<typename T>
T f(T x) {
  (void)(x + 0);
  (void)T(0);
  (void)(x += 0);
  (void)(x? x : x);
  return g(x);
  //  h(x); // h is a dependent name
  return 0;
}
