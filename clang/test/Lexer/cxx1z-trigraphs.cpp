// RUN: %clang_cc1 -std=c++1z %s -verify
// RUN: %clang_cc1 -std=c++1z %s -trigraphs -fsyntax-only

??= define foo ; // expected-error {{}} expected-warning {{trigraph ignored}}

static_assert("??="[0] == '#', ""); // expected-error {{failed}} expected-warning {{trigraph ignored}}

// ??/
error here; // expected-error {{}}
