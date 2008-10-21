// RUN: clang -fsyntax-only -pedantic -verify %s 
int* quals1(int const * p);
int* quals2(int const * const * pp);
int* quals3(int const * * const * ppp);

void test_quals(int * p, int * * pp, int * * * ppp) {
  int const * const * pp2 = pp; 
  quals1(p);
  quals2(pp);
  quals3(ppp); // expected-error {{ incompatible type passing 'int ***', expected 'int const **const *' }}
}
