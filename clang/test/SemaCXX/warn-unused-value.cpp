// RUN: %clang_cc1 -fsyntax-only -verify -Wunused-value %s

// PR4806
namespace test0 {
  class Box {
  public:
    int i;
    volatile int j;
  };

  void doit() {
    // pointer to volatile has side effect (thus no warning)
    Box* box = new Box;
    box->i; // expected-warning {{expression result unused}}
    box->j;
  }
}
