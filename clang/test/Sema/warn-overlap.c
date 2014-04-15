// RUN: %clang_cc1 -fsyntax-only -verify -Wtautological-overlap-compare %s

#define mydefine 2

void f(int x) {
  int y = 0;

  // > || <
  if (x > 2 || x < 1) { }
  if (x > 2 || x < 2) { }
  if (x != 2 || x != 3) { } // expected-warning {{overlapping comparisons always evaluate to true}}
  if (x > 2 || x < 3) { } // expected-warning {{overlapping comparisons always evaluate to true}}
  if (x > 0 || x < 0) { }

  if (x > 2 || x <= 1) { }
  if (x > 2 || x <= 2) { } // expected-warning {{overlapping comparisons always evaluate to true}}
  if (x > 2 || x <= 3) { } // expected-warning {{overlapping comparisons always evaluate to true}}

  if (x >= 2 || x < 1) { }
  if (x >= 2 || x < 2) { } // expected-warning {{overlapping comparisons always evaluate to true}}
  if (x >= 2 || x < 3) { } // expected-warning {{overlapping comparisons always evaluate to true}}

  if (x >= 2 || x <= 1) { } // expected-warning {{overlapping comparisons always evaluate to true}}
  if (x >= 2 || x <= 2) { } // expected-warning {{overlapping comparisons always evaluate to true}}
  if (x >= 2 || x <= 3) { } // expected-warning {{overlapping comparisons always evaluate to true}}
  if (x >= 0 || x <= 0) { } // expected-warning {{overlapping comparisons always evaluate to true}}

  // > && <
  if (x > 2 && x < 1) { }  // expected-warning {{overlapping comparisons always evaluate to false}}
  if (x > 2 && x < 2) { }  // expected-warning {{overlapping comparisons always evaluate to false}}
  if (x > 2 && x < 3) { }  // expected-warning {{overlapping comparisons always evaluate to false}}
  if (x > 0 && x < 1) { }  // expected-warning {{overlapping comparisons always evaluate to false}}

  if (x > 2 && x <= 1) { }  // expected-warning {{overlapping comparisons always evaluate to false}}
  if (x > 2 && x <= 2) { }  // expected-warning {{overlapping comparisons always evaluate to false}}
  if (x > 2 && x <= 3) { }

  if (x >= 2 && x < 1) { }  // expected-warning {{overlapping comparisons always evaluate to false}}
  if (x >= 2 && x < 2) { }  // expected-warning {{overlapping comparisons always evaluate to false}}
  if (x >= 2 && x < 3) { }
  if (x >= 0 && x < 0) { }  // expected-warning {{overlapping comparisons always evaluate to false}}

  if (x >= 2 && x <= 1) { }  // expected-warning {{overlapping comparisons always evaluate to false}}
  if (x >= 2 && x <= 2) { }
  if (x >= 2 && x <= 3) { }

  // !=, ==, ..
  if (x != 2 || x != 3) { }  // expected-warning {{overlapping comparisons always evaluate to true}}
  if (x != 2 || x < 3) { }   // expected-warning {{overlapping comparisons always evaluate to true}}
  if (x == 2 && x == 3) { }  // expected-warning {{overlapping comparisons always evaluate to false}}
  if (x == 2 && x > 3) { }   // expected-warning {{overlapping comparisons always evaluate to false}}
  if (x == 3 && x < 0) { }  // expected-warning {{overlapping comparisons always evaluate to false}}
  if (3 == x && x < 0) { }  // expected-warning {{overlapping comparisons always evaluate to false}}

  if (x == mydefine && x > 3) { }
  if (x == (mydefine + 1) && x > 3) { }
}

// Don't generate a warning here.
void array_out_of_bounds() {
  int x;
  int buffer[4];
  x = (-7 > 0) ? (buffer[-7]) : 0;
}
