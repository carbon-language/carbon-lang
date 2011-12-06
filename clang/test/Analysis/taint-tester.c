// RUN: %clang_cc1  -analyze -analyzer-checker=experimental.security.taint,debug.TaintTest -verify %s

int scanf(const char *restrict format, ...);
int getchar(void);

#define BUFSIZE 10
int Buffer[BUFSIZE];

void bufferScanfAssignment(int x) {
  int n;
  int *addr = &Buffer[0];
  scanf("%d", &n);
  addr += n;// expected-warning {{tainted}}
  *addr = n; // expected-warning 2 {{tainted}}

  double tdiv = n / 30; // expected-warning 3 {{tainted}}
  char *loc_cast = (char *) n; // expected-warning {{tainted}}
  char tinc = tdiv++; // expected-warning {{tainted}}
  int tincdec = (char)tinc--; // expected-warning 2 {{tainted}}
  int tprtarithmetic1 = *(addr+1);


}
