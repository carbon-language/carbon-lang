// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace A {
  class String; // expected-note {{target of using declaration}}
};

using A::String; // expected-note {{using declaration}}
class String; // expected-error {{conflicts with target of using declaration}}

// rdar://8603569
union value {
char *String;
};
