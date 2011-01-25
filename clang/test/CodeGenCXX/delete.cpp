// RUN: %clang_cc1 -triple x86_64-apple-darwin10 %s -emit-llvm -o - | FileCheck %s

void t1(int *a) {
  delete a;
}

struct S {
  int a;
};

// POD types.
void t3(S *s) {
  delete s;
}

// Non-POD
struct T {
  ~T();
  int a;
};

// CHECK: define void @_Z2t4P1T
void t4(T *t) {
  // CHECK: call void @_ZN1TD1Ev
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: call void @_ZdlPv
  delete t;
}

// PR5102
template <typename T>
class A {
  operator T *() const;
};

void f() {
  A<char*> a;
  
  delete a;
}

namespace test0 {
  struct A {
    void *operator new(__SIZE_TYPE__ sz);
    void operator delete(void *p) { ::operator delete(p); }
    ~A() {}
  };

  // CHECK: define void @_ZN5test04testEPNS_1AE(
  void test(A *a) {
    // CHECK: call void @_ZN5test01AD1Ev
    // CHECK-NEXT: bitcast
    // CHECK-NEXT: call void @_ZN5test01AdlEPv
    delete a;
  }

  // CHECK: define linkonce_odr void @_ZN5test01AD1Ev(%class.A* %this) unnamed_addr
  // CHECK: define linkonce_odr void @_ZN5test01AdlEPv
}

namespace test1 {
  struct A {
    int x;
    ~A();
  };

  // CHECK: define void @_ZN5test14testEPA10_A20_NS_1AE(
  void test(A (*arr)[10][20]) {
    delete [] arr;
    // CHECK:      icmp eq [10 x [20 x [[S:%.*]]]]* [[PTR:%.*]], null
    // CHECK-NEXT: br i1

    // CHECK:      [[ARR:%.*]] = getelementptr inbounds [10 x [20 x [[S]]]]* [[PTR]], i32 0, i32 0, i32 0
    // CHECK-NEXT: bitcast {{.*}} to i8*
    // CHECK-NEXT: [[ALLOC:%.*]] = getelementptr inbounds {{.*}}, i64 -8
    // CHECK-NEXT: bitcast i8* [[ALLOC]] to i64*
    // CHECK-NEXT: load
    // CHECK-NEXT: store i64 {{.*}}, i64* [[IDX:%.*]]

    // CHECK:      load i64* [[IDX]]
    // CHECK-NEXT: icmp ne {{.*}}, 0
    // CHECK-NEXT: br i1

    // CHECK:      load i64* [[IDX]]
    // CHECK-NEXT: [[I:%.*]] = sub i64 {{.*}}, 1
    // CHECK-NEXT: getelementptr inbounds [[S]]* [[ARR]], i64 [[I]]
    // CHECK-NEXT: call void @_ZN5test11AD1Ev(
    // CHECK-NEXT: br label

    // CHECK:      load i64* [[IDX]]
    // CHECK-NEXT: sub
    // CHECK-NEXT: store {{.*}}, i64* [[IDX]]
    // CHECK-NEXT: br label

    // CHECK:      call void @_ZdaPv(i8* [[ALLOC]])
  }
}

namespace test2 {
  // CHECK: define void @_ZN5test21fEPb
  void f(bool *b) {
    // CHECK: call void @_ZdlPv(i8*
    delete b;
    // CHECK: call void @_ZdaPv(i8*
    delete [] b;
  }
}

namespace test3 {
  void f(int a[10][20]) {
    // CHECK: call void @_ZdaPv(i8*
    delete a;
  }
}
