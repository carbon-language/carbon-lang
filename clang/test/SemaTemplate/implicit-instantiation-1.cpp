// RUN: clang-cc -fsyntax-only -verify %s

template<typename T, typename U>
struct X {
  T f(T x, U y) { return x + y; }

  unsigned g(T x, U y) { return sizeof(f(x, y)); }
};

void test(X<int, int> *xii, X<int*, int> *xpi, X<int, int*> *xip) {
  (void)xii->f(1, 2);
  (void)xpi->f(0, 2);
  (void)sizeof(xip->f(2, 0)); // okay: does not instantiate
  (void)xip->g(2, 0); // okay: does not instantiate
}

