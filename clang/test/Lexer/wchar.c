// RUN: %clang_cc1 -fsyntax-only -fshort-wchar -verify %s

void f() {
  (void)L"\U00010000";  // expected-warning {{character unicode escape sequence too long for its type}}

  (void)L'\U00010000';  // expected-warning {{character unicode escape sequence too long for its type}}

  (void)L'ab';  // expected-warning {{extraneous characters in wide character constant ignored}}

  (void)L'a\u1000';  // expected-warning {{extraneous characters in wide character constant ignored}}
}

