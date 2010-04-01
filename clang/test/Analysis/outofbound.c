// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -analyzer-experimental-checks -analyzer-check-objc-mem -analyzer-store=region -verify %s

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

struct three_words {
  int c[3];
};

struct seven_words {
  int c[7];
};

void f3() {
  struct three_words a, *p;
  p = &a;
  p[0] = a; // no-warning
  p[1] = a; // expected-warning{{Access out-of-bound array element (buffer overflow)}}
}

void f4() {
  struct seven_words c;
  struct three_words a, *p = (struct three_words *)&c;
  p[0] = a; // no-warning
  p[1] = a; // no-warning
  p[2] = a; // expected-warning{{Access out-of-bound array element (buffer overflow)}}
}
