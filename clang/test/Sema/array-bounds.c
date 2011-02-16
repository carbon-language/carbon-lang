// RUN: %clang_cc1 -verify %s

int foo() {
  int x[2];
  int y[2];
  int *p = &y[2]; // no-warning
  (void) sizeof(x[2]); // no-warning
  y[2] = 2; // expected-warning{{array index excedes last array element}}
  return x[2] +  // expected-warning{{array index excedes last array element}}
         y[-1] + // expected-warning{{array index precedes first array element}}
         x[sizeof(x)] +  // expected-warning{{array index excedes last array element}}
         x[sizeof(x) / sizeof(x[0])] +  // expected-warning{{array index excedes last array element}}
         x[sizeof(x) / sizeof(x[0]) - 1] + // no-warning
         x[sizeof(x[2])]; // expected-warning{{array index excedes last array element}}
}

