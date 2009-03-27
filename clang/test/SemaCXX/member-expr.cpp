// RUN: clang-cc -fsyntax-only -verify %s

class X{
public:
  enum E {Enumerator};
  int f();
  static int mem;
  static float g();
};

void test(X* xp, X x) {
  int i1 = x.f();
  int i2 = xp->f();
  x.E; // expected-error{{cannot refer to type member 'E' with '.'}}
  xp->E; // expected-error{{cannot refer to type member 'E' with '->'}}
  int i3 = x.Enumerator;
  int i4 = xp->Enumerator;
  x.mem = 1;
  xp->mem = 2;
  float f1 = x.g();
  float f2 = xp->g();
}

struct A {
 int f0;
};
struct B {
 A *f0();
};
int f0(B *b) {
  return b->f0->f0; // expected-error{{member reference base type 'struct A *(void)' is not a structure or union}} \
  // expected-note{{perhaps you meant to call this function}}
}
