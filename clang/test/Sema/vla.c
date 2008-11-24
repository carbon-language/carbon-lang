// RUN: clang %s -verify -fsyntax-only

int test1() {
  typedef int x[test1()];  // vla
  static int y = sizeof(x);  // expected-error {{not a compile-time constant}}
}

// PR2347
void f (unsigned int m)
{
  int e[2][m];

  e[0][0] = 0;
}

// PR3048
int x = sizeof(struct{char qq[x];}); // expected-error {{fields must have a constant size}}

