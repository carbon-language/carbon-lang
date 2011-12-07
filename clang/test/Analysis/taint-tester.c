// RUN: %clang_cc1  -analyze -analyzer-checker=experimental.security.taint,debug.TaintTest -verify %s

int scanf(const char *restrict format, ...);
int getchar(void);

#define BUFSIZE 10
int Buffer[BUFSIZE];

struct XYStruct {
  int x;
  float y;
};

void taintTracking(int x) {
  int n;
  int *addr = &Buffer[0];
  scanf("%d", &n);
  addr += n;// expected-warning 2 {{tainted}}
  *addr = n; // expected-warning 3 {{tainted}}

  double tdiv = n / 30; // expected-warning 3 {{tainted}}
  char *loc_cast = (char *) n; // expected-warning {{tainted}}
  char tinc = tdiv++; // expected-warning {{tainted}}
  int tincdec = (char)tinc--; // expected-warning 2 {{tainted}}

  // Tainted ptr arithmetic/array element address.
  int tprtarithmetic1 = *(addr+1); // expected-warning 2 {{tainted}}

  // Tainted struct address, casts.
  struct XYStruct *xyPtr = 0;
  scanf("%p", &xyPtr);
  void *tXYStructPtr = xyPtr; // expected-warning 2 {{tainted}}
  struct XYStruct *xyPtrCopy = tXYStructPtr; // expected-warning 2 {{tainted}}
}
