// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -I %S/Inputs -fmodule-cache-path %t %s -verify


// in other file: expected-note{{previous definition is here}}





// in other file: expected-note{{previous definition is here}}

@import decldef;
A *a1; // expected-error{{unknown type name 'A'}}
B *b1; // expected-error{{unknown type name 'B'}}
@import decldef.Decl;

A *a2;
B *b;

void testA(A *a) {
  a->ivar = 17; // expected-error{{definition of 'A' must be imported before it is required}}
}

void testB() {
  B b; // expected-error{{definition of 'B' must be imported before it is required}}
  B b2; // Note: the reundant error was silenced.
}
