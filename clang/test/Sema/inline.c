// RUN: %clang_cc1 -std=gnu89 -fsyntax-only -verify %s

// Check that we don't allow illegal uses of inline
inline int a; // expected-error{{'inline' can only appear on functions}}
typedef inline int b; // expected-error{{'inline' can only appear on functions}}
int d(inline int a); // expected-error{{'inline' can only appear on functions}}

// PR5253
// GNU Extension: check that we can redefine an extern inline function
extern inline int f(int a) {return a;}
int f(int b) {return b;} // expected-note{{previous definition is here}}
// And now check that we can't redefine a normal function
int f(int c) {return c;} // expected-error{{redefinition of 'f'}}

// Check that we can redefine an extern inline function as a static function
extern inline int g(int a) {return a;}
static int g(int b) {return b;}

// Check that we ensure the types of the two definitions are the same
extern inline int h(int a) {return a;} // expected-note{{previous definition is here}}
int h(short b) {return b;}  // expected-error{{conflicting types for 'h'}}

