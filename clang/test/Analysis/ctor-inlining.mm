// RUN: %clang_cc1 -analyze -analyzer-checker=core -fobjc-arc -cfg-add-implicit-dtors -Wno-null-dereference -verify %s

struct Wrapper {
  __strong id obj;
};

void test() {
  Wrapper w;
  // force a diagnostic
  *(char *)0 = 1; // expected-warning{{Dereference of null pointer}}
}
