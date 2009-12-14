// RUN: clang -cc1 -analyze -warn-uninit-values -verify %s

int f1() {
  int x;
  return x; // expected-warning {{use of uninitialized variable}}
}

int f2(int x) {
  int y;
  int z = x + y; // expected-warning {{use of uninitialized variable}}
  return z;
}


int f3(int x) {
  int y;
  return x ? 1 : y; // expected-warning {{use of uninitialized variable}}
}

int f4(int x) {
  int y;
  if (x) y = 1;
  return y; // expected-warning {{use of uninitialized variable}}
}

void f5() {
  int a;
  a = 30; // no-warning
}

void f6(int i) {
  int x;
  for (i = 0 ; i < 10; i++)
    printf("%d",x++); // expected-warning {{use of uninitialized variable}} \
  // expected-warning{{implicitly declaring C library function 'printf' with type 'int (char const *, ...)'}} \
  // expected-note{{please include the header <stdio.h> or explicitly provide a declaration for 'printf'}}
}

void f7(int i) {
  int x = i;
  int y;
  for (i = 0; i < 10; i++ ) {
    printf("%d",x++); // no-warning
    x += y; // expected-warning {{use of uninitialized variable}}
  }
}

int f8(int j) {
  int x = 1, y = x + 1;
  if (y) // no-warning
    return x;
  return y;
}
