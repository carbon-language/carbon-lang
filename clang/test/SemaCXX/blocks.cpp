// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s -fblocks

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

// rdar: // 8382559
namespace radar8382559 {
  void func(bool& outHasProperty);

  int test3() {
    __attribute__((__blocks__(byref))) bool hasProperty = false;
    bool has = true;

    bool (^b)() = ^ {
     func(hasProperty);
     if (hasProperty)
       hasProperty = 0;
     if (has)
       hasProperty = 1;
     return hasProperty;
     };
    func(hasProperty);
    func(has);
    b();
    if (hasProperty)
      hasProperty = 1;
    if (has)
      has = 2;
    return hasProperty = 1;
  }
}

// Move __block variables to the heap when possible.
class MoveOnly {
public:
  MoveOnly();
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly(MoveOnly&&);
};

void move_block() {
  __block MoveOnly mo;
}

// Don't crash after failing to build a block due to a capture of an
// invalid declaration.
namespace test5 {
  struct B { // expected-note 2 {{candidate constructor}}
    void *p;
    B(int); // expected-note {{candidate constructor}}
  };

  void use_block(void (^)());
  void use_block_2(void (^)(), const B &a);

  void test() {
    B x; // expected-error {{no matching constructor for initialization}}
    use_block(^{
        int y;
        use_block_2(^{ (void) y; }, x);
      });
  }
}
