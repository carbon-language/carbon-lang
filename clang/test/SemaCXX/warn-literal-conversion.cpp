// RUN: %clang_cc1 -fsyntax-only -Wliteral-conversion -verify %s

void foo(int y);

// Warn when a literal float or double is assigned or bound to an integer.
void test0() {
  // Float
  int y0 = 1.2222F; // expected-warning {{implicit conversion turns literal floating-point number into integer}}
  int y1 = (1.2222F); // expected-warning {{implicit conversion turns literal floating-point number into integer}}
  int y2 = (((1.2222F))); // expected-warning {{implicit conversion turns literal floating-point number into integer}}
  int y3 = 12E1F; // expected-warning {{implicit conversion turns literal floating-point number into integer}}
  int y4 = 1.2E1F; // expected-warning {{implicit conversion turns literal floating-point number into integer}}
  // Double
  int y5 = 1.2222; // expected-warning {{implicit conversion turns literal floating-point number into integer}}
  int y6 = 12E1; // expected-warning {{implicit conversion turns literal floating-point number into integer}}
  int y7 = 1.2E1; // expected-warning {{implicit conversion turns literal floating-point number into integer}}
  int y8 = (1.2E1); // expected-warning {{implicit conversion turns literal floating-point number into integer}}

  // Test assignment to an existing variable.
  y8 = 2.22F; // expected-warning {{implicit conversion turns literal floating-point number into integer}}

  // Test direct initialization.
  int y9(1.23F); // expected-warning {{implicit conversion turns literal floating-point number into integer}}

  // Test passing a literal floating-point value to a function that takes an integer.
  foo(1.2F); // expected-warning {{implicit conversion turns literal floating-point number into integer}}

  // FIXME: -Wconversion-literal doesn't catch "-1.2F".
  int y10 = -1.2F;

  // -Wconversion-literal does NOT catch const values.
  // (-Wconversion DOES catch them.)
  static const float sales_tax_rate = .095F;
  int z = sales_tax_rate;
  foo(sales_tax_rate);

  // Expressions, such as those that indicate rounding-down, should NOT produce warnings.
  int x = 24 * 0.5;
  int y = (24*60*60) * 0.25;
  int pennies = 123.45 * 100;
}
