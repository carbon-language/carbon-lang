// RUN: %clang_cc1 -fsyntax-only -verify -Wshorten-64-to-32 -triple x86_64-apple-darwin %s

int test0(long v) {
  return v; // expected-warning {{implicit conversion loses integer precision}}
}
