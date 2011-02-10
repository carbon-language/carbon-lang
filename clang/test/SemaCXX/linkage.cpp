// This is an IR generation test because the calculation of visibility
// during IR gen will cause linkage to be implicitly recomputed and
// compared against the earlier cached value.  If we had a way of
// testing linkage directly in Sema, that would be better.

// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck %s

// PR8926
namespace test0 {
  typedef struct {
    void *foo() { return 0; }
  } A;

  // CHECK: define linkonce_odr i8* @_ZN5test01A3fooEv(

  void test(A *a) {
    a->foo();
  }
}

namespace test1 {
  typedef struct {
    template <unsigned n> void *foo() { return 0; }

    void foo() {
      foo<0>();
    }
  } A;

  // CHECK: define linkonce_odr void @_ZN5test11A3fooEv(
  // another at the end

  void test(A *a) {
    a->foo();
  }
}

namespace test2 {
  typedef struct {
    template <unsigned n> struct B {
      void *foo() { return 0; }
    };

    void foo(B<0> *b) {
      b->foo();
    }
  } A;

  // CHECK: define linkonce_odr void @_ZN5test21A3fooEPNS0_1BILj0EEE(

  void test(A *a) {
    a->foo(0);
  }
}

namespace test3 {
  namespace { struct A {}; }

  // CHECK: define internal void @_ZN5test34testENS_12_GLOBAL__N_11AE(
  void test(A a) {}
  void force() { test(A()); }

  // CHECK: define void @test3(
  extern "C" void test3(A a) {}
}

// CHECK: define linkonce_odr i8* @_ZN5test21A1BILj0EE3fooEv(
// CHECK: define linkonce_odr i8* @_ZN5test11A3fooILj0EEEPvv(
