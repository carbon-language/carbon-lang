// RUN: clang-cc -analyze -checker-cfref -analyzer-store=region -verify %s
// XFAIL: *

char f1() {
  char* s = "abcd";
  char c = s[4]; // no-warning
  return s[5] + c; // expected-warning{{Load or store into an out-of-bound memory position.}}
}
