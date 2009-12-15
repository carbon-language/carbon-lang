// RUN: %clang_cc1 -Wparentheses -fsyntax-only -verify %s
// RUN: %clang_cc1 -Wparentheses -fixit %s -o - | %clang_cc1 -Wparentheses -Werror -

// Test the various warnings under -Wparentheses
void if_assign(void) {
  int i;
  if (i = 4) {} // expected-warning {{assignment as a condition}}
  if ((i = 4)) {}
}

void bitwise_rel(unsigned i) {
  (void)(i & 0x2 == 0); // expected-warning {{& has lower precedence than ==}}
  (void)(0 == i & 0x2); // expected-warning {{& has lower precedence than ==}}
  (void)(i & 0xff < 30); // expected-warning {{& has lower precedence than <}}
  (void)((i & 0x2) == 0);
  (void)(i & (0x2 == 0));
  // Eager logical op
  (void)(i == 1 | i == 2 | i == 3);
  (void)(i != 1 & i != 2 & i != 3);
}
