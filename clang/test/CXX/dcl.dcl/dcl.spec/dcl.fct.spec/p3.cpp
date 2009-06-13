// RUN: clang-cc -verify %s
// XFAIL

void f0(void) {
  inline void f1(); // expected-error {{'inline' is not allowed on block scope function declaration}}
}

// FIXME: Add test for "If the inline specifier is used in a friend declaration,
// that declaration shall be a definition or the function shall have previously
// been declared inline.

