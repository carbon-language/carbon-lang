// RUN: clang-cc -fsyntax-only -verify %s

struct A {};
enum B { Dummy };
namespace C {}
struct D : A {};
struct E : A {};
struct F : D, E {};
struct G : virtual D {};

int A::*pdi1;
int (::A::*pdi2);
int (A::*pfi)(int);

int B::*pbi; // expected-error {{expected a class or namespace}} \
             // expected-error{{does not point into a class}}
int C::*pci; // expected-error {{'pci' does not point into a class}}
void A::*pdv; // expected-error {{'pdv' declared as a member pointer to void}}
int& A::*pdr; // expected-error {{'pdr' declared as a member pointer to a reference}}

void f() {
  // This requires tentative parsing.
  int (A::*pf)(int, int);

  // Implicit conversion to bool.
  bool b = pdi1;
  b = pfi;

  // Conversion from null pointer constant.
  pf = 0;
  pf = __null;

  // Conversion to member of derived.
  int D::*pdid = pdi1;
  pdid = pdi2;

  // Fail conversion due to ambiguity and virtuality.
  int F::*pdif = pdi1; // expected-error {{ambiguous conversion from pointer to member of base class 'struct A' to pointer to member of derived class 'struct F'}} expected-error {{incompatible type}}
  int G::*pdig = pdi1; // expected-error {{conversion from pointer to member of class 'struct A' to pointer to member of class 'struct G' via virtual base 'struct D' is not allowed}} expected-error {{incompatible type}}

  // Conversion to member of base.
  pdi1 = pdid; // expected-error {{incompatible type assigning 'int struct D::*', expected 'int struct A::*'}}
  
  // Comparisons
  int (A::*pf2)(int, int);
  int (D::*pf3)(int, int) = 0;
  bool b1 = (pf == pf2); (void)b1;
  bool b2 = (pf != pf2); (void)b2;
  bool b3 = (pf == pf3); (void)b3;
  bool b4 = (pf != 0); (void)b4;
}

struct TheBase
{
  void d();
};

struct HasMembers : TheBase
{
  int i;
  void f();

  void g();
  void g(int);
  static void g(double);
};

namespace Fake
{
  int i;
  void f();
}

void g() {
  HasMembers hm;

  int HasMembers::*pmi = &HasMembers::i;
  int *pni = &Fake::i;
  int *pmii = &hm.i;

  void (HasMembers::*pmf)() = &HasMembers::f;
  void (*pnf)() = &Fake::f;
  &hm.f; // FIXME: needs diagnostic expected-warning{{result unused}}

  void (HasMembers::*pmgv)() = &HasMembers::g;
  void (HasMembers::*pmgi)(int) = &HasMembers::g;
  void (*pmgd)(double) = &HasMembers::g;

  void (HasMembers::*pmd)() = &HasMembers::d;
}

struct Incomplete;

void h() {
  HasMembers hm, *phm = &hm;

  int HasMembers::*pi = &HasMembers::i;
  hm.*pi = 0;
  int i = phm->*pi;
  (void)&(hm.*pi);
  (void)&(phm->*pi);
  (void)&((&hm)->*pi); 

  void (HasMembers::*pf)() = &HasMembers::f;
  (hm.*pf)();
  (phm->*pf)();

  (void)(hm->*pi); // expected-error {{left hand operand to ->* must be a pointer to class compatible with the right hand operand, but is 'struct HasMembers'}}
  (void)(phm.*pi); // expected-error {{left hand operand to .* must be a class compatible with the right hand operand, but is 'struct HasMembers *'}}
  (void)(i.*pi); // expected-error {{left hand operand to .* must be a class compatible with the right hand operand, but is 'int'}}
  int *ptr;
  (void)(ptr->*pi); // expected-error {{left hand operand to ->* must be a pointer to class compatible with the right hand operand, but is 'int *'}}

  int A::*pai = 0;
  D d, *pd = &d;
  (void)(d.*pai);
  (void)(pd->*pai);
  F f, *ptrf = &f;
  (void)(f.*pai); // expected-error {{left hand operand to .* must be a class compatible with the right hand operand, but is 'struct F'}}
  (void)(ptrf->*pai); // expected-error {{left hand operand to ->* must be a pointer to class compatible with the right hand operand, but is 'struct F *'}}

  (void)(hm.*i); // expected-error {{pointer-to-member}}
  (void)(phm->*i); // expected-error {{pointer-to-member}}

  Incomplete *inc;
  int Incomplete::*pii = 0;
  (void)(inc->*pii); // okay
}

struct OverloadsPtrMem
{
  int operator ->*(const char *);
};

void i() {
  OverloadsPtrMem m;
  int foo = m->*"Awesome!";
}
