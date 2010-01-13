// RUN: %clang_cc1 -fsyntax-only -Wunreachable-code -fblocks -verify %s

int j;
void bar() { }
int test1() {
  for (int i = 0;
       i != 10;
       ++i) {  // expected-warning {{will never be executed}}
    if (j == 23) // missing {}'s
      bar();
      return 1;
  }
  return 0;
  return 1;    // expected-warning {{will never be executed}}
}

void test2(int i) {
  switch (i) {
  case 0:
    break;
    bar();     // expected-warning {{will never be executed}}
  case 2:
    switch (i) {
    default:
    a: goto a;
    }
    bar();     // expected-warning {{will never be executed}}
  }
  b: goto b;
  bar();       // expected-warning {{will never be executed}}
}

void test3() {
  ^{ return;
     bar();    // expected-warning {{will never be executed}}
  }();
  while (++j) {
    continue;
    bar();     // expected-warning {{will never be executed}}
  }
}
