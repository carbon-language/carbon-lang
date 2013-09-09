// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fobjc-arc -I %S/Inputs -fmodules-cache-path=%t %s -verify -DUSE_EARLY
// RUN: %clang_cc1 -fmodules -fobjc-arc -I %S/Inputs -fmodules-cache-path=%t %s -verify

// expected-note@Inputs/def.h:5 {{previous}}

@class Def;
Def *def;
class Def2; // expected-note {{forward decl}}
Def2 *def2;
namespace Def3NS { class Def3; } // expected-note {{forward decl}}
Def3NS::Def3 *def3;

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
  // FIXME: These should both work, since we've (implicitly) imported
  // decldef.Def here, but they don't, because nothing has triggered the lazy
  // loading of the definitions of these classes.
  def2->func(); // expected-error {{incomplete}}
  def3->func(); // expected-error {{incomplete}}
}
