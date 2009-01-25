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

struct A {};
void mquals1(int const A::*p);
void mquals2(int const A::* const A::*pp);
void mquals3(int const A::* A::* const A::*ppp);

void test_mquals(int A::*p, int A::* A::*pp, int A::* A::* A::*ppp) {
  int const A::* const A::* pp2 = pp;
  mquals1(p);
  mquals2(pp);
  mquals3(ppp); // expected-error {{ incompatible type passing 'int struct A::*struct A::*struct A::*', expected 'int const struct A::*struct A::*const struct A::*' }}
}
