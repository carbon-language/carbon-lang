// RUN: clang %s -fsyntax-only -verify -pedantic

typedef struct S S;
void a(S* b, void* c) {
  b++;       // expected-error {{arithmetic on pointer to incomplete type}}
  b += 1;    // expected-error {{arithmetic on pointer to incomplete type}}
  c++;       // expected-warning {{use of GNU void* extension}}
  c += 1;    // expected-warning {{use of GNU void* extension}}
  b = 1+b;   // expected-error {{arithmetic on pointer to incomplete type}}
  /* The next couple tests are only pedantic warnings in gcc */
  void (*d)(S*,void*) = a;
  d += 1;    // expected-error {{pointer to incomplete type}}
  d++;       // expected-error {{pointer to incomplete type}}
}
