// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

int i [[clang::lto_visibility_public]]; // expected-warning {{'lto_visibility_public' attribute only applies to structs, unions, and classes}}
typedef int t [[clang::lto_visibility_public]]; // expected-warning {{'lto_visibility_public' attribute only applies to}}
[[clang::lto_visibility_public]] void f(); // expected-warning {{'lto_visibility_public' attribute only applies to}}
void f() [[clang::lto_visibility_public]]; // expected-error {{'lto_visibility_public' attribute cannot be applied to types}}

struct [[clang::lto_visibility_public]] s1 {
  int i [[clang::lto_visibility_public]]; // expected-warning {{'lto_visibility_public' attribute only applies to}}
  [[clang::lto_visibility_public]] void f(); // expected-warning {{'lto_visibility_public' attribute only applies to}}
};

struct [[clang::lto_visibility_public(1)]] s2 { // expected-error {{'lto_visibility_public' attribute takes no arguments}}
};
