// RUN: %clang_cc1 %s -triple x86_64-linux -emit-llvm -o - -mconstructor-aliases | FileCheck %s

namespace test1 {
// test that we produce an alias when the destructor is weak_odr

// CHECK-DAG: @_ZN5test16foobarIvEC1Ev = alias weak_odr void (%"struct.test1::foobar"*)* @_ZN5test16foobarIvEC2Ev
// CHECK-DAG: define weak_odr void @_ZN5test16foobarIvEC2Ev(
template <typename T> struct foobar {
  foobar() {}
};

template struct foobar<void>;
}

namespace test2 {
// test that when the destrucor is linkonce_odr we just replace every use of
// C1 with C2.

// CHECK-DAG: define linkonce_odr void @_ZN5test26foobarIvEC2Ev(
// CHECK-DAG: call void @_ZN5test26foobarIvEC2Ev
void g();
template <typename T> struct foobar {
  foobar() { g(); }
};
foobar<void> x;
}

namespace test3 {
// test that these alias are internal.

// CHECK-DAG: @_ZN5test312_GLOBAL__N_11AD1Ev = alias internal void (%"struct.test3::<anonymous namespace>::A"*)* @_ZN5test312_GLOBAL__N_11AD2Ev
// CHECK-DAG: @_ZN5test312_GLOBAL__N_11BD2Ev = alias internal bitcast (void (%"struct.test3::<anonymous namespace>::A"*)* @_ZN5test312_GLOBAL__N_11AD2Ev to void (%"struct.test3::<anonymous namespace>::B"*)*)
// CHECK-DAG: @_ZN5test312_GLOBAL__N_11BD1Ev = alias internal void (%"struct.test3::<anonymous namespace>::B"*)* @_ZN5test312_GLOBAL__N_11BD2Ev
// CHECK-DAG: define internal void @_ZN5test312_GLOBAL__N_11AD2Ev(
namespace {
struct A {
  ~A() {}
};

struct B : public A {};
}

B x;
}

namespace test4 {
  // Test that we don't produce aliases from B to A. We cannot because we cannot
  // guarantee that they will be present in every TU. Instead, we just call
  // A's destructor directly.

  // CHECK-DAG: define linkonce_odr void @_ZN5test41AD2Ev(
  // CHECK-DAG: call i32 @__cxa_atexit{{.*}}_ZN5test41AD2Ev
  struct A {
    virtual ~A() {}
  };
  struct B : public A{
    ~B() {}
  };
  B X;
}
