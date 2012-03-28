// RUN: %clang_cc1 %s -triple x86_64-apple-darwin10 -emit-llvm -verify -o - |FileCheck %s

class x {
public: int operator=(int);
};
void a() {
  x a;
  a = 1u;
}

void f(int i, int j) {
  // CHECK: load i32
  // CHECK: load i32
  // CHECK: add nsw i32
  // CHECK: store i32
  // CHECK: store i32 17, i32
  // CHECK: ret
  (i += j) = 17;
}

// Taken from g++.old-deja/g++.jason/net.C
namespace test1 {
  template <class T> void fn (T t) { }
  template <class T> struct A {
    void (*p)(T);
    A() { p = fn; }
  };

  A<int> a;
}

// PR12204
namespace test2 {
  struct A {
    A() {} // make this non-POD to enable tail layout
    void *ptr;
    char c;
  };

  void test(A &out) {
    out = A();
  }
}
// CHECK:    define void @_ZN5test24testERNS_1AE(
// CHECK:      [[OUT:%.*]] = alloca [[A:%.*]]*, align 8
// CHECK-NEXT: [[TMP:%.*]] = alloca [[A]], align 8
// CHECK:      [[REF:%.*]] = load [[A]]** [[OUT]], align 8
// CHECK-NEXT: call void @_ZN5test21AC1Ev([[A]]* [[TMP]])
// CHECK-NEXT: [[T0:%.*]] = bitcast [[A]]* [[REF]] to i8*
// CHECK-NEXT: [[T1:%.*]] = bitcast [[A]]* [[TMP]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* [[T0]], i8* [[T1]], i64 9, i32 8, i1 false)
