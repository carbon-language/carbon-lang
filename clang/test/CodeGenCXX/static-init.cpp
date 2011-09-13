// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s

// CHECK: @_ZZ1hvE1i = internal global i32 0, align 4

// CHECK: @_ZZN5test1L6getvarEiE3var = internal constant [4 x i32] [i32 1, i32 0, i32 2, i32 4], align 16
// CHECK: @_ZZ2h2vE1i = linkonce_odr global i32 0
// CHECK: @_ZGVZ2h2vE1i = linkonce_odr global i64 0

struct A {
  A();
  ~A();
};

void f() {
  // CHECK: load atomic i8* bitcast (i64* @_ZGVZ1fvE1a to i8*) acquire, align 1
  // CHECK: call i32 @__cxa_guard_acquire
  // CHECK: call void @_ZN1AC1Ev
  // CHECK: call i32 @__cxa_atexit(void (i8*)* bitcast (void (%struct.A*)* @_ZN1AD1Ev to void (i8*)*), i8* getelementptr inbounds (%struct.A* @_ZZ1fvE1a, i32 0, i32 0), i8* bitcast (i8** @__dso_handle to i8*))
  // CHECK: call void @__cxa_guard_release
  static A a;
}

void g() {
  // CHECK: call noalias i8* @_Znwm(i64 1)
  // CHECK: call void @_ZN1AC1Ev(
  static A& a = *new A;
}

int a();
void h() {
  static const int i = a();
}

inline void h2() {
  static int i = a();
}

void h3() {
  h2();
}

// PR6980: this shouldn't crash
namespace test0 {
  struct A { A(); };
  __attribute__((noreturn)) int throw_exception();

  void test() {
    throw_exception();
    static A r;
  }
}

namespace test1 {
  // CHECK: define internal i32 @_ZN5test1L6getvarEi(
  static inline int getvar(int index) {
    static const int var[] = { 1, 0, 2, 4 };
    return var[index];
  }

  void test() { (void) getvar(2); }
}
