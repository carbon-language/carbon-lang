// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-apple-darwin %s

// This file tests -Wconstant-conversion, a subcategory of -Wconversion
// which is on by default.

// rdar://problem/6792488
void test_6792488(void) {
  int x = 0x3ff0000000000000U; // expected-warning {{implicit conversion from 'unsigned long' to 'int' changes value from 4607182418800017408 to 0}}
}

void test_7809123(void) {
  struct { int i5 : 5; } a;

  a.i5 = 36; // expected-warning {{implicit truncation from 'int' to bitfield changes value from 36 to 4}}
}

void test() {
  struct { int bit : 1; } a;
  a.bit = 1; // shouldn't warn
}

enum Test2 { K_zero, K_one };
enum Test2 test2(enum Test2 *t) {
  *t = 20;
  return 10; // shouldn't warn
}
