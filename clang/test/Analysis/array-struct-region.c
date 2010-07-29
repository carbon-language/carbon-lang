// RUN: %clang_cc1 -analyze -analyzer-experimental-checks -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-store=region -analyzer-constraints=basic -verify %s
// RUN: %clang_cc1 -analyze -analyzer-experimental-checks -analyzer-experimental-internal-checks -analyzer-check-objc-mem -analyzer-store=region -analyzer-constraints=range -verify %s

int string_literal_init() {
  char a[] = "abc";
  char b[2] = "abc"; // expected-warning{{too long}}
  char c[5] = "abc";

  if (a[1] != 'b')
    return 0; // expected-warning{{never executed}}
  if (b[1] != 'b')
    return 0; // expected-warning{{never executed}}
  if (c[1] != 'b')
    return 0; // expected-warning{{never executed}}

  if (a[3] != 0)
    return 0; // expected-warning{{never executed}}
  if (c[3] != 0)
    return 0; // expected-warning{{never executed}}

  if (c[4] != 0)
    return 0; // expected-warning{{never executed}}

  return 42;
}
