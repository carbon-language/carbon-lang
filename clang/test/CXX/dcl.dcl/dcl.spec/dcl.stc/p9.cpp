// RUN: %clang_cc1 -verify %s

struct S; // expected-note {{forward declaration of 'S'}}
extern S a;
extern S f(); // expected-note {{'f' declared here}}
extern void g(S a); // expected-note {{candidate function}}

void h() {
  // FIXME: This diagnostic could be better.
  g(a); // expected-error {{no matching function for call to 'g'}}
  f(); // expected-error {{calling 'f' with incomplete return type 'S'}}
}
