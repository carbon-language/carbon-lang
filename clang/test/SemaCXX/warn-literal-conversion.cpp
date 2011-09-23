// RUN: %clang_cc1 -fsyntax-only -Wliteral-conversion -verify %s

void foo(int y);

// Warn when a literal float or double is assigned or bound to an integer.
void test0() {
  // Float
  int y0 = 1.2222F; // expected-warning {{implicit conversion turns literal floating-point number into integer}}
  int y1 = (1.2222F); // expected-warning {{implicit conversion turns literal floating-point number into integer}}
  int y2 = (((1.2222F))); // expected-warning {{implicit conversion turns literal floating-point number into integer}}
  int y3 = 12E1F; // expected-warning {{implicit conversion turns literal floating-point number into integer}} \
                  // expected-note {{this can be rewritten as an integer literal with the exact same value}}
  int y4 = 1.2E1F; // expected-warning {{implicit conversion turns literal floating-point number into integer}} \
                   // expected-note {{this can be rewritten as an integer literal with the exact same value}}
  // Double
  int y5 = 1.2222; // expected-warning {{implicit conversion turns literal floating-point number into integer}}
  int y6 = 12E1; // expected-warning {{implicit conversion turns literal floating-point number into integer}} \
                 // expected-note {{this can be rewritten as an integer literal with the exact same value}}
  int y7 = 1.2E1; // expected-warning {{implicit conversion turns literal floating-point number into integer}} \
                  // expected-note {{this can be rewritten as an integer literal with the exact same value}}
  int y8 = (1.2E1); // expected-warning {{implicit conversion turns literal floating-point number into integer}} \
                    // expected-note {{this can be rewritten as an integer literal with the exact same value}}

  // Test assignment to an existing variable.
  y8 = 2.22F; // expected-warning {{implicit conversion turns literal floating-point number into integer}}

  // Test direct initialization.
  int y9(1.23F); // expected-warning {{implicit conversion turns literal floating-point number into integer}}

  // Test passing a literal floating-point value to a function that takes an integer.
  foo(1.2F); // expected-warning {{implicit conversion turns literal floating-point number into integer}}

  int y10 = -1.2F;  // expected-warning {{implicit conversion turns literal floating-point number into integer}}

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

// Warn on cases where a string literal is converted into a bool.
// An exception is made for this in logical operators.
void assert(bool condition);
void test1() {
  bool b0 = "hi"; // expected-warning{{implicit conversion turns string literal into bool: 'const char [3]' to 'bool'}}
  b0 = ""; // expected-warning{{implicit conversion turns string literal into bool: 'const char [1]' to 'bool'}}
  b0 = 0 && "";
  assert("error"); // expected-warning{{implicit conversion turns string literal into bool: 'const char [6]' to 'bool'}}
  assert(0 && "error");

  while("hi") {} // expected-warning{{implicit conversion turns string literal into bool: 'const char [3]' to 'bool'}}
  do {} while("hi"); // expected-warning{{implicit conversion turns string literal into bool: 'const char [3]' to 'bool'}}
  for (;"hi";); // expected-warning{{implicit conversion turns string literal into bool: 'const char [3]' to 'bool'}}
  if("hi") {} // expected-warning{{implicit conversion turns string literal into bool: 'const char [3]' to 'bool'}}
}
