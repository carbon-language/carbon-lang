// RUN: %clang_cc1 -fsyntax-only -verify %s

template <class Element>
void foo() {
}
template <>
 __attribute__((visibility("hidden")))  // expected-note {{previous attribute is here}}
void foo<int>();

template <>
void foo<int>();

template <>
 __attribute__((visibility("default"))) // expected-error {{visibility does not match previous declaration}}
void foo<int>() {
}

struct x3 {
  static int y;
} __attribute((visibility("default"))); // expected-warning {{attribute 'visibility' after definition is ignored}}
