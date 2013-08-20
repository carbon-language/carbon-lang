// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fobjc-arc -I %S/Inputs -fmodules-cache-path=%t %s -verify -DUSE_EARLY
// RUN: %clang_cc1 -fmodules -fobjc-arc -I %S/Inputs -fmodules-cache-path=%t %s -verify

// expected-note@Inputs/def.h:5 {{previous}}

@class Def;
Def *def;
class Def2;
Def2 *def2;

@interface Unrelated
- defMethod;
@end

@import decldef;
#ifdef USE_EARLY
A *a1; // expected-error{{declaration of 'A' must be imported from module 'decldef.Def'}}
B *b1;
#endif
@import decldef.Decl;

A *a2;
B *b;

void testA(A *a) {
  a->ivar = 17;
#ifndef USE_EARLY
  // expected-error@-2{{definition of 'A' must be imported from module 'decldef.Def' before it is required}}
#endif
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
