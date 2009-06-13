// RUN: clang-cc -verify %s

struct S; // expected-note {{forward declaration of 'struct S'}}
extern S a;
extern S f();
extern void g(S a); // expected-note {{candidate function}}

void h() {
  // FIXME: This diagnostic could be better.
  g(a); // expected-error {{no matching function for call to 'g'}}
  f(); // expected-error {{return type of called function ('struct S') is incomplete}}
}
