// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -warn-dead-stores -verify %s
// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=basic -analyzer-constraints=basic -warn-dead-stores -verify %s
// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=basic -analyzer-constraints=range -warn-dead-stores -verify %s
// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -analyzer-constraints=basic -warn-dead-stores -verify %s
// RUN: %clang_cc1 -analyze -analyzer-experimental-internal-checks -checker-cfref -analyzer-store=region -analyzer-constraints=range -warn-dead-stores -verify %s

//===----------------------------------------------------------------------===//
// Basic dead store checking (but in C++ mode).
//===----------------------------------------------------------------------===//

int j;
void f1() {
  int x = 4;

  ++x; // expected-warning{{never read}}

  switch (j) {
  case 1:
    throw 1;
    (void)x;
    break;
  }
}

//===----------------------------------------------------------------------===//
// Dead store checking involving constructors.
//===----------------------------------------------------------------------===//

class Test1 {
  int &x;
public:
  Test1(int &y) : x(y) {}
  ~Test1() { ++x; }
};

int test_ctor_1(int x) {
  { Test1 a(x); } // no-warning
  return x;
}
