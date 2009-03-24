// RUN: clang-cc -analyze -checker-simple -analyzer-store=region -verify %s

char f1() {
  char* s = "abcd";
  char c = s[4]; // no-warning
  return s[5] + c; // expected-warning{{Load or store into an out-of-bound memory position.}}
}
