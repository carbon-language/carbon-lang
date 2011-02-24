// RUN: %clang_cc1 -analyze -analyzer-experimental-checks -analyzer-checker=core.experimental.UnreachableCode -analyzer-check-objc-mem -analyzer-store=region -analyzer-constraints=basic -verify %s
// RUN: %clang_cc1 -analyze -analyzer-experimental-checks -analyzer-checker=core.experimental.UnreachableCode -analyzer-check-objc-mem -analyzer-store=region -analyzer-constraints=range -verify %s

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

void nested_compound_literals(int rad) {
  int vec[6][2] = {{0.195, 0.02}, {0.383, 0.067}, {0.55, 0.169},
                   {0.831, 0.45}, {0.924, 0.617}, {0.98, 0.805}};
  int a;

  for (a = 0; a < 6; ++a) {
      vec[a][0] *= rad; // no-warning
      vec[a][1] *= rad; // no-warning
  }
}

void nested_compound_literals_float(float rad) {
  float vec[6][2] = {{0.195, 0.02}, {0.383, 0.067}, {0.55, 0.169},
                     {0.831, 0.45}, {0.924, 0.617}, {0.98, 0.805}};
  int a;

  for (a = 0; a < 6; ++a) {
      vec[a][0] *= rad; // no-warning
      vec[a][1] *= rad; // no-warning
  }
}
