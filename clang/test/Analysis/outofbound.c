// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -analyzer-experimental-checks -analyzer-check-objc-mem -analyzer-store=region -verify %s

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void *calloc(size_t, size_t);

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

void f5() {
  char *p = calloc(2,2);
  p[3] = '.'; // no-warning
  p[4] = '!'; // expected-warning{{out-of-bound}}
}

void f6() {
  char a[2];
  int *b = (int*)a;
  b[1] = 3; // expected-warning{{out-of-bound}}
}

void f7() {
  struct three_words a;
  a.c[3] = 1; // expected-warning{{out-of-bound}}
}

void vla(int a) {
  if (a == 5) {
    int x[a];
    x[4] = 4; // no-warning
    x[5] = 5; // expected-warning{{out-of-bound}}
  }
}

void sizeof_vla(int a) {
  if (a == 5) {
    char x[a];
    int y[sizeof(x)];
    y[4] = 4; // no-warning
    y[5] = 5; // expected-warning{{out-of-bound}}
  }
}

void alloca_region(int a) {
  if (a == 5) {
    char *x = __builtin_alloca(a);
    x[4] = 4; // no-warning
    x[5] = 5; // expected-warning{{out-of-bound}}
  }
}

int symbolic_index(int a) {
  int x[2] = {1, 2};
  if (a == 2) {
    return x[a]; // expected-warning{{out-of-bound}}
  }
  return 0;
}

int symbolic_index2(int a) {
  int x[2] = {1, 2};
  if (a < 0) {
    return x[a]; // expected-warning{{out-of-bound}}
  }
  return 0;
}
