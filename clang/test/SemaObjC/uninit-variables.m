// RUN: %clang_cc1 -fsyntax-only -Wuninitialized -fsyntax-only -fblocks %s -verify

// Duplicated from uninit-variables.c.
// Test just to ensure the analysis is working.
int test1() {
  int x; // expected-note{{variable 'x' is declared here}} expected-note{{add initialization}}
  return x; // expected-warning{{variable 'x' is uninitialized when used here}}
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

