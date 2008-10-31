// RUN: clang -fsyntax-only -verify %s

struct A {};
struct B : A {};
struct C : B {};

struct D : private A {};
struct E : A {};
struct F : B, E {};

struct Incomplete;

void basic_bad()
{
  // ptr -> nonptr
  (void)dynamic_cast<A>((A*)0); // expected-error {{'struct A' is not a reference or pointer}}
  // nonptr -> ptr
  (void)dynamic_cast<A*>(0); // expected-error {{'int' is not a pointer}}
  // ptr -> noncls
  (void)dynamic_cast<int*>((A*)0); // expected-error {{'int' is not a class}}
  // noncls -> ptr
  (void)dynamic_cast<A*>((int*)0); // expected-error {{'int' is not a class}}
  // ref -> noncls
  (void)dynamic_cast<int&>(*((A*)0)); // expected-error {{'int' is not a class}}
  // noncls -> ref
  (void)dynamic_cast<A&>(*((int*)0)); // expected-error {{'int' is not a class}}
  // ptr -> incomplete
  (void)dynamic_cast<Incomplete*>((A*)0); // expected-error {{'struct Incomplete' is incomplete}}
  // incomplete -> ptr
  (void)dynamic_cast<A*>((Incomplete*)0); // expected-error {{'struct Incomplete' is incomplete}}
}

void same()
{
  (void)dynamic_cast<A*>((A*)0);
  (void)dynamic_cast<A&>(*((A*)0));
}

void up()
{
  (void)dynamic_cast<A*>((B*)0);
  (void)dynamic_cast<A&>(*((B*)0));
  (void)dynamic_cast<A*>((C*)0);
  (void)dynamic_cast<A&>(*((C*)0));

  // Inaccessible
  //(void)dynamic_cast<A*>((D*)0);
  //(void)dynamic_cast<A&>(*((D*)0));

  // Ambiguous
  (void)dynamic_cast<A*>((F*)0); // expected-error {{ambiguous conversion from derived class 'struct F' to base class 'struct A':\n    struct F -> struct B -> struct A\n    struct F -> struct E -> struct A}}
  (void)dynamic_cast<A&>(*((F*)0)); // expected-error {{ambiguous conversion from derived class 'struct F' to base class 'struct A':\n    struct F -> struct B -> struct A\n    struct F -> struct E -> struct A}}
}

// FIXME: Other test cases require recognition of polymorphic classes.
