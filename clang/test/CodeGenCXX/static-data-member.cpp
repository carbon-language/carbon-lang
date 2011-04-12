// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm -o - %s | FileCheck %s

// CHECK: @_ZN5test11A1aE = constant i32 10, align 4
// CHECK: @_ZN5test212_GLOBAL__N_11AIiE1xE = internal global i32 0, align 4
// CHECK: @_ZN5test31AIiE1xE = weak_odr global i32 0, align 4
// CHECK: @_ZGVN5test31AIiE1xE = weak_odr global i64 0

// PR5564.
namespace test1 {
  struct A {
    static const int a = 10;
  };

  const int A::a;

  struct S { 
    static int i;
  };

  void f() { 
    int a = S::i;
  }
}

// Test that we don't use guards for initializing template static data
// members with internal linkage.
namespace test2 {
  int foo();

  namespace {
    template <class T> struct A {
      static int x;
    };

    template <class T> int A<T>::x = foo();
    template struct A<int>;
  }

  // CHECK: define internal void @__cxx_global_var_init()
  // CHECK:      [[TMP:%.*]] = call i32 @_ZN5test23fooEv()
  // CHECK-NEXT: store i32 [[TMP]], i32* @_ZN5test212_GLOBAL__N_11AIiE1xE, align 4
  // CHECK-NEXT: ret void
}

// Test that we don't use threadsafe statics when initializing
// template static data members.
namespace test3 {
  int foo();

  template <class T> struct A {
    static int x;
  };

  template <class T> int A<T>::x = foo();
  template struct A<int>;

  // CHECK: define internal void @__cxx_global_var_init1()
  // CHECK:      [[GUARDBYTE:%.*]] = load i8* bitcast (i64* @_ZGVN5test31AIiE1xE to i8*)
  // CHECK-NEXT: [[UNINITIALIZED:%.*]] = icmp eq i8 [[GUARDBYTE]], 0
  // CHECK-NEXT: br i1 [[UNINITIALIZED]]
  // CHECK:      [[TMP:%.*]] = call i32 @_ZN5test33fooEv()
  // CHECK-NEXT: store i32 [[TMP]], i32* @_ZN5test31AIiE1xE, align 4
  // CHECK-NEXT: store i64 1, i64* @_ZGVN5test31AIiE1xE
  // CHECK-NEXT: br label
  // CHECK:      ret void
}
