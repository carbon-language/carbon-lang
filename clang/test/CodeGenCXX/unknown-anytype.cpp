// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -funknown-anytype -emit-llvm -o - %s | FileCheck %s

int test0() {
  extern __unknown_anytype test0_any;
  // CHECK: load i32* @test0_any
  return (int) test0_any;
}

int test1() {
  extern __unknown_anytype test1_any;
  // CHECK: call i32 @test1_any()
  return (int) test1_any();
}

float test2() {
  extern __unknown_anytype test2_any;
  // CHECK: call float @test2_any(float {{[^,]+}})
  return (float) test2_any(0.5f);
}

float test3() {
  extern __unknown_anytype test3_any;
  // CHECK: call float @test3_any(i32 5)
  return ((float(int)) test3_any)(5);
}

namespace test4 {
  extern __unknown_anytype test4_any1;
  extern __unknown_anytype test4_any2;

  int test() {
    // CHECK: load i32* @_ZN5test410test4_any1E
    // CHECK: call i32 @_ZN5test410test4_any2E
    return (int) test4_any1 + (int) test4_any2();
  }
}

void test5() {
  extern __unknown_anytype test5_any;
  // CHECK: call void @test5_any()
  return (void) test5_any();
}

long test6() {
  extern __unknown_anytype test6_any(float *);
  // CHECK: call i64 @_Z9test6_anyPf(float* null)
  return (long) test6_any(0);
}

struct Test7 {
  ~Test7();
};
Test7 test7() {
  extern __unknown_anytype test7_any;
  // CHECK: call void @test7_any({{%.*}}* sret {{%.*}}, i32 5)
  return (Test7) test7_any(5);
}

struct Test8 {
  __unknown_anytype foo();
  __unknown_anytype foo(int);

  void test();
};
void Test8::test() {
  (int) foo();
  (int) foo(5);
  (float) this->foo();
  (float) this->foo(5);
}
void test8(Test8 *p) {
  (double) p->foo();
  (double) p->foo(5);
  (bool) (*p).foo();
  (bool) (*p).foo(5);
}
