// RUN: clang-cc -analyze -checker-cfref -analyzer-store=region -verify %s

char f1() {
  char* s = "abcd";
  char c = s[4]; // no-warning
  return s[5] + c; // expected-warning{{Access out-of-bound array element (buffer overflow)}}
}
