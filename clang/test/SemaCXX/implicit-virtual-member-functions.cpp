// RUN: %clang_cc1 -fsyntax-only -triple %itanium_abi_triple -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple %ms_abi_triple -DMSABI -verify %s
struct A {
  virtual ~A();
};

struct B : A { // expected-error {{no suitable member 'operator delete' in 'B'}}
  virtual void f();

  void operator delete (void *, int); // expected-note {{'operator delete' declared here}}
};

#ifdef MSABI
B b; // expected-note {{implicit destructor for 'B' first required here}}
#else
void B::f() { // expected-note {{implicit destructor for 'B' first required here}}
}
#endif

struct C : A { // expected-error {{no suitable member 'operator delete' in 'C'}}
  C();
  void operator delete(void *, int); // expected-note {{'operator delete' declared here}}
};

C::C() { }  // expected-note {{implicit destructor for 'C' first required here}}

struct D : A { // expected-error {{no suitable member 'operator delete' in 'D'}}
  void operator delete(void *, int); // expected-note {{'operator delete' declared here}}
}; 

void f() {
  new D; // expected-note {{implicit destructor for 'D' first required here}}
}
