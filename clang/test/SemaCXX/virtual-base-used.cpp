// RUN: %clang_cc1 -fsyntax-only -triple %itanium_abi_triple -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple %ms_abi_triple -DMSABI -verify %s
// PR7800

// The Microsoft ABI doesn't have the concept of key functions, so we have different
// expectations about when functions are first required for that case.

#ifdef MSABI
// expected-note@+2 3 {{declared private here}}
#endif
class NoDestroy { ~NoDestroy(); }; // expected-note 3 {{declared private here}}
struct A {
  virtual ~A();
};

#ifdef MSABI
// expected-error@+3 {{field of type 'NoDestroy' has private destructor}}
#endif
struct B : public virtual A {
  NoDestroy x; // expected-error {{field of type 'NoDestroy' has private destructor}}
};
#ifdef MSABI
// expected-note@+3 {{implicit default constructor for 'B' first required here}}
// expected-note@+2 {{implicit destructor for 'B' first required here}}
#endif
struct D : public virtual B {
  virtual void foo();
  ~D();
};
#ifdef MSABI
D d; // expected-note {{implicit default constructor for 'D' first required here}}
#else
void D::foo() { // expected-note {{implicit destructor for 'B' first required here}}
}
#endif

#ifdef MSABI
// expected-error@+3 {{field of type 'NoDestroy' has private destructor}}
#endif
struct E : public virtual A {
  NoDestroy x; // expected-error {{field of type 'NoDestroy' has private destructor}}
};
#ifdef MSABI
// expected-note@+2 {{implicit default constructor for 'E' first required here}}
#endif
struct F : public E { // expected-note {{implicit destructor for 'E' first required here}}
};
#ifdef MSABI
// expected-note@+2 {{implicit default constructor for 'F' first required here}}
#endif
struct G : public virtual F {
  virtual void foo();
  ~G();
};
#ifdef MSABI
G g; // expected-note {{implicit default constructor for 'G' first required here}}
#else
void G::foo() { // expected-note {{implicit destructor for 'F' first required here}}
}
#endif

#ifdef MSABI
// expected-note@+3 {{'H' declared here}}
// expected-error@+3 {{field of type 'NoDestroy' has private destructor}}
#endif
struct H : public virtual A {
  NoDestroy x; // expected-error {{field of type 'NoDestroy' has private destructor}}
};
#ifdef MSABI
// expected-error@+3 {{implicit default constructor for 'I' must explicitly initialize the base class 'H' which does not have a default constructor}}
// expected-note@+2 {{implicit destructor for 'H' first required here}}
#endif
struct I : public virtual H {
  ~I();
};
#ifdef MSABI
// expected-note@+3 {{implicit default constructor for 'H' first required here}}
// expected-note@+2 {{implicit default constructor for 'I' first required here}}
#endif
struct J : public I {
  virtual void foo();
  ~J();
};
#ifdef MSABI
J j; // expected-note {{implicit default constructor for 'J' first required here}}
#else
void J::foo() { // expected-note {{implicit destructor for 'H' first required here}}
}
#endif
