// RUN: %clang_cc1 -analyze -analyzer-experimental-checks -analyzer-check-objc-mem -analyzer-check-dead-stores -verify -analyzer-opt-analyze-nested-blocks %s

extern void foo(int a);

void test(unsigned a) {
  switch (a) {
    a += 5; // expected-warning{{never executed}}
  case 2:
    a *= 10;
  case 3:
    a %= 2;
  }
  foo(a);
}

void test2(unsigned a) {
 help:
  if (a > 0)
    return;
  if (a == 0)
    return;
  foo(a); // expected-warning{{never executed}}
  goto help;
}

void test3() {
  int a = 5;

  while (a > 1)
    a -= 2;

  if (a > 1) {
    a = a + 56; // expected-warning{{never executed}}
  }

  foo(a);
}

void test4(unsigned a) {
  while(1);
  if (a > 5) { // expected-warning{{never executed}}
    return;
  }
}

extern void bar(char c);

void test5(const char *c) {
  foo(c[0]);

  if (!c) {
    bar(1); // expected-warning{{never executed}}
  }
}

void test6(const char *c) {
  if (c) return;
  if (!c) return;
  __builtin_unreachable(); // no-warning
}

