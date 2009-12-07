// RUN: clang-cc -fsyntax-only -verify %s
struct A {
  virtual ~A();
};

struct B : A {  // expected-error {{no suitable member 'operator delete' in 'B'}}
  B() { }  // expected-note {{implicit default destructor for 'struct B' first required here}}
  void operator delete(void *, int); // expected-note {{'operator delete' declared here}}
};

struct C : A {  // expected-error {{no suitable member 'operator delete' in 'C'}}
  void operator delete(void *, int); // expected-note {{'operator delete' declared here}}
};

void f() {
  // new B should mark the constructor as used, which then marks
  // all the virtual members as used, because B has no key function.
  (void)new B;

  // Same here, except that C has an implicit constructor.
  (void)new C; // expected-note {{implicit default destructor for 'struct C' first required here}}
}
