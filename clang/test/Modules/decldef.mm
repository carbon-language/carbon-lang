// RUN: rm -rf %t
// RUN: %clang_cc1 -I %S/Inputs -fmodule-cache-path %t %s -verify

__import_module__ decldef;
A *a1; // expected-error{{unknown type name 'A'}}
B *b1; // expected-error{{unknown type name 'B'}}

__import_module__ decldef.Decl;

// FIXME: No link between @interface (which we can't see) and @class
// (which we can).
// A *a2;
B *b;

void testB() {
  B b; // FIXME: Should error, because we can't see the definition.
}
