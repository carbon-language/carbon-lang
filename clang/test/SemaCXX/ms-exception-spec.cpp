// RUN: %clang_cc1 %s -fsyntax-only -verify -fms-extensions

void f() throw(...) { }

namespace PR28080 {
struct S; // expected-note {{forward declaration}}
void fn() throw(S); // expected-warning {{incomplete type}}
}
