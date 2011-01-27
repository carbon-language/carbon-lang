// RUN: %clang_cc1 -fsyntax-only -Wuninitialized -fsyntax-only -fblocks %s -verify

// Duplicated from uninit-variables.c.
// Test just to ensure the analysis is working.
int test1() {
  int x; // expected-warning{{use of uninitialized variable 'x'}} expected-note{{add initialization to silence this warning}}
  return x; // expected-note{{variable 'x' is possibly uninitialized when used here}}
}

// Test ObjC fast enumeration.
void test2() {
  id collection = 0;
  for (id obj in collection) {
    if (0 == obj) // no-warning
      break;
  }
}

void test3() {
  id collection = 0;
  id obj;
  for (obj in collection) { // no-warning
    if (0 == obj) // no-warning
      break;
  }
}

