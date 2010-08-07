// RUN: %clang_cc1 -fsyntax-only -verify %s
// PR7800

class NoDestroy { ~NoDestroy(); }; // expected-note {{declared private here}}
struct A {
  virtual ~A();
};
struct B : public virtual A {
  NoDestroy x; // expected-error {{field of type 'NoDestroy' has private destructor}}
};
struct D : public virtual B {
  virtual void foo();
  ~D();
};
void D::foo() { // expected-note {{implicit default destructor for 'B' first required here}}
}
