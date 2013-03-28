// RUN: %clang_cc1 -analyze -analyzer-checker=core,cplusplus.NewDelete -analyzer-store region -std=c++11 -fblocks -verify %s
// expected-no-diagnostics

namespace std {
  typedef __typeof__(sizeof(int)) size_t;
}

void *operator new(std::size_t, ...);
void *operator new[](std::size_t, ...);

void testGlobalCustomVariadicNew() {
  void *p1 = operator new(0); // no warn

  void *p2 = operator new[](0); // no warn

  int *p3 = new int; // no warn

  int *p4 = new int[0]; // no warn
}
