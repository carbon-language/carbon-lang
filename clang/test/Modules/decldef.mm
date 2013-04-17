// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fobjc-arc -I %S/Inputs -fmodules-cache-path=%t %s -verify

// expected-note@Inputs/def.h:5 {{previous definition is here}}

@class Def;
Def *def;
class Def2;
Def2 *def2;

@interface Unrelated
- defMethod;
@end

@import decldef;
A *a1; // expected-error{{unknown type name 'A'}}
B *b1; // expected-error{{unknown type name 'B'}}
@import decldef.Decl;

A *a2;
B *b;

void testA(A *a) {
  a->ivar = 17; // expected-error{{definition of 'A' must be imported from module 'decldef.Def' before it is required}}
}

void testB() {
  B b; // Note: redundant error silenced
}

void testDef() {
  [def defMethod];
}

void testDef2() {
  def2->func();
}
