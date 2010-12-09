// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s

struct A {
  union {
    int a;
    void* b;
  };
  
  A() : a(0) { }
};

A a;

namespace PR7021 {
  struct X
  {
    union { long l; };
  };

  // CHECK: define void @_ZN6PR70211fENS_1XES0_
  void f(X x, X z) {
    X x1;

    // CHECK: store i64 1, i64
    x1.l = 1;

    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
    X x2(x1);

    X x3;
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
    x3 = x1;

    // CHECK: ret void
  }
}

namespace test2 {
  struct A {
    struct {
      union {
        int b;
      };
    };

    A();
  };

  A::A() : b(10) { }
  // CHECK: define void @_ZN5test21AC2Ev(
  // CHECK-NOT: }
  // CHECK: store i32 10
  // CHECK: }
}

namespace test3 {
  struct A {
    union {
      mutable char fibers[100];
      struct {
        void (*callback)(void*);
        void *callback_value;
      };
    };

    A();
  };

  A::A() : callback(0), callback_value(0) {}
  // CHECK: define void @ZN5test31AC2Ev(
  // CHECK: [[THIS:%.*]] = load
  // CHECK-NEXT: [[UNION:%.*]] = getelementptr inbounds {{.*}} [[THIS]], i32 0, i32 0
  // CHECK-NEXT: [[STRUCT:%.*]] = getelementptr inbounds {{.*}} [[UNION]], i32 0, i32 0
  // CHECK-NEXT: [[CALLBACK:%.*]] = getelementptr inbounds {{.*}} [[STRUCT]], i32 0, i32 0
  // CHECK-NEXT: store void (i8*)* null, void (i8*)** [[CALLBACK]]
  // CHECK-NEXT: [[UNION:%.*]] = getelementptr inbounds {{.*}} [[THIS]], i32 0, i32 0
  // CHECK-NEXT: [[STRUCT:%.*]] = getelementptr inbounds {{.*}} [[UNION]], i32 0, i32 0
  // CHECK-NEXT: [[CVALUE:%.*]] = getelementptr inbounds {{.*}} [[STRUCT]], i32 0, i32 0
  // CHECK-NEXT: store i8* null, void i8** [[CVALUE]]
}

struct S {
  // CHECK: store i32 42
  // CHECK: store i32 55
  S() : x(42), y(55) {}
  union {
    struct {
      int x;
      union { int y; };
    };
  };
} s;


 //PR8760 
 template <typename T>
  struct Foo {
    Foo() : ptr(__nullptr) {}
    union {
      T *ptr;
   };
  };
  Foo<int> f;
  
  
