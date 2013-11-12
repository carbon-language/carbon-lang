// RUN: %clang_cc1 %s -triple x86_64-linux -emit-llvm -o - -mconstructor-aliases | FileCheck %s

namespace test1 {
// test that we don't produce an alias when the destructor is weak_odr. The
// reason to avoid it that another TU might have no explicit template
// instantiation definition or declaration, causing it to to output only
// one of the destructors as linkonce_odr, producing a different comdat.

// CHECK-DAG: define weak_odr void @_ZN5test16foobarIvEC2Ev
// CHECK-DAG: define weak_odr void @_ZN5test16foobarIvEC1Ev

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
// test that instead of an internal alias we just use the other destructor
// directly.

// CHECK-DAG: define internal void @_ZN5test312_GLOBAL__N_11AD2Ev(
// CHECK-DAG: call i32 @__cxa_atexit{{.*}}_ZN5test312_GLOBAL__N_11AD2Ev
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

namespace test5 {
  // similar to test4, but with an internal B.

  // CHECK-DAG: define linkonce_odr void @_ZN5test51AD2Ev(
  // CHECK-DAG: call i32 @__cxa_atexit{{.*}}_ZN5test51AD2Ev
  struct A {
    virtual ~A() {}
  };
  namespace {
  struct B : public A{
    ~B() {}
  };
  }
  B X;
}

namespace test6 {
  // Test that we use ~A directly, even when ~A is not defined. The symbol for
  // ~B would have been internal and still contain a reference to ~A.
  struct A {
    virtual ~A();
  };
  namespace {
  struct B : public A {
    ~B() {}
  };
  }
  B X;
  // CHECK-DAG: call i32 @__cxa_atexit({{.*}}@_ZN5test61AD2Ev
}

namespace test7 {
  // Test that we don't produce an alias from ~B to ~A<int> (or crash figuring
  // out if we should).
  // pr17875.
  // CHECK-DAG: define void @_ZN5test71BD2Ev
  template <typename> struct A {
    ~A() {}
  };
  class B : A<int> {
    ~B();
  };
  template class A<int>;
  B::~B() {}
}

namespace test8 {
  // Test that we replace ~zed with ~bar which is an alias to ~foo.
  // CHECK-DAG: call i32 @__cxa_atexit({{.*}}@_ZN5test83barD2Ev
  // CHECK-DAG: @_ZN5test83barD2Ev = alias {{.*}} @_ZN5test83fooD2Ev
  struct foo {
    ~foo();
  };
  foo::~foo() {}
  struct bar : public foo {
    ~bar();
  };
  bar::~bar() {}
  struct zed : public bar {};
  zed foo;
}
