// RUN: clang -fsyntax-only -verify %s

struct A {};
enum B { Dummy };
namespace C {}
struct D : A {};

int A::*pdi1;
int (::A::*pdi2);
int (A::*pfi)(int);

int B::*pbi; // expected-error {{expected a class or namespace}}
int C::*pci; // expected-error {{'pci' does not point into a class}}
void A::*pdv; // expected-error {{'pdv' declared as a member pointer to void}}
int& A::*pdr; // expected-error {{'pdr' declared as a pointer to a reference}}

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
}
