// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -analyzer-experimental-checks -checker-cfref -analyzer-store=region -verify %s

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);

char f1() {
  char* s = "abcd";
  char c = s[4]; // no-warning
  return s[5] + c; // expected-warning{{Access out-of-bound array element (buffer overflow)}}
}

void f2() {
  int *p = malloc(12);
  p[3] = 4; // expected-warning{{Access out-of-bound array element (buffer overflow)}}
}
