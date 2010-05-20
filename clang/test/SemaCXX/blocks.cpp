// RUN: %clang_cc1 -fsyntax-only -verify %s -fblocks

void tovoid(void*);

void tovoid_test(int (^f)(int, int)) {
  tovoid(f);
}

void reference_lvalue_test(int& (^f)()) {
  f() = 10;
}

// PR 7165
namespace test1 {
  void g(void (^)());
  struct Foo {
    void foo();   
    void test() {
      (void) ^{ foo(); };
    }
  };
}

namespace test2 {
  int repeat(int value, int (^block)(int), unsigned n) {
    while (n--) value = block(value);
    return value;
  }

  class Power {
    int base;

  public:
    Power(int base) : base(base) {}
    int calculate(unsigned n) {
      return repeat(1, ^(int v) { return v * base; }, n);
    }
  };

  int test() {
    return Power(2).calculate(10);
  }
}
