// RUN: %clang_cc1 -verify %s

int foo() {
  int x[2]; // expected-note 4 {{array 'x' declared here}}
  int y[2]; // expected-note 2 {{array 'y' declared here}}
  int *p = &y[2]; // no-warning
  (void) sizeof(x[2]); // no-warning
  y[2] = 2; // expected-warning{{array index of '2' indexes past the end of an array (that contains 2 elements)}}
  return x[2] +  // expected-warning{{array index of '2' indexes past the end of an array (that contains 2 elements)}}
         y[-1] + // expected-warning{{array index of '-1' indexes before the beginning of the array}}
         x[sizeof(x)] +  // expected-warning{{array index of '8' indexes past the end of an array (that contains 2 elements)}}
         x[sizeof(x) / sizeof(x[0])] +  // expected-warning{{array index of '2' indexes past the end of an array (that contains 2 elements)}}
         x[sizeof(x) / sizeof(x[0]) - 1] + // no-warning
         x[sizeof(x[2])]; // expected-warning{{array index of '4' indexes past the end of an array (that contains 2 elements)}}
}

// This code example tests that -Warray-bounds works with arrays that
// are template parameters.
template <char *sz> class Qux {
  bool test() { return sz[0] == 'a'; }
};