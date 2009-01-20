// RUN: clang -analyze -checker-simple -analyzer-store-region -verify %s

char f1() {
  char* s = "abcd";
  return s[5]; // expected-warning{{Load or store into an out-of-bound memory position.}}
}
