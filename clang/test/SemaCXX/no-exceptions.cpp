// RUN: %clang_cc1 -fsyntax-only -verify %s

// Various tests for -fno-exceptions

typedef __SIZE_TYPE__ size_t;

namespace test0 {
  // rdar://problem/7878149
  class Foo {
  public:
    void* operator new(size_t x);
  private:
    void operator delete(void *x);
  };

  void test() {
    // Under -fexceptions, this does access control for the associated
    // 'operator delete'.
    (void) new Foo();
  }
}
