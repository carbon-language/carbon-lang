// RUN: clang -fsyntax-only -verify -std=c++0x %s

typedef int&& irr;
typedef irr& ilr_c1; // Collapses to int&
typedef int& ilr;
typedef ilr&& ilr_c2; // Collapses to int&

irr ret_irr() {
  return 0;
}

struct not_int {};

int over(int&);
not_int over(int&&);

void f() {
  int &&virr1; // expected-error {{declaration of reference variable 'virr1' requires an initializer}}
  int &&virr2 = 0;
  // FIXME: named rvalue references are lvalues!
  //int &&virr3 = virr1; // xpected-error {{rvalue reference cannot bind to lvalue}}
  int i1 = 0;
  int &&virr4 = i1; // expected-error {{rvalue reference cannot bind to lvalue}}
  int &&virr5 = ret_irr();

  int i2 = over(i1);
  not_int ni1 = over(0);
  int i3 = over(virr2);
  not_int ni2 = over(ret_irr());

  ilr_c1 vilr1 = i1;
  ilr_c2 vilr2 = i1;
}
