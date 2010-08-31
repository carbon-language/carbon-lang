// RUN: %clang_cc1 %s -triple=thumbv7-apple-darwin3.0.0-iphoneos -fno-use-cxa-atexit -target-abi apcs-gnu -emit-llvm -o - -fexceptions | FileCheck %s

class foo {
public:
    foo();
    virtual ~foo();
};

class bar : public foo {
public:
	bar();
};

// The global dtor needs the right calling conv with -fno-use-cxa-atexit
// rdar://7817590
// Checked at end of file.
bar baz;

// Destructors and constructors must return this.
namespace test1 {
  void foo();

  struct A {
    A(int i) { foo(); }
    ~A() { foo(); }
    void bar() { foo(); }
  };

  // CHECK: define void @_ZN5test14testEv()
  void test() {
    // CHECK: [[AV:%.*]] = alloca [[A:%.*]], align 1
    // CHECK: call [[A]]* @_ZN5test11AC1Ei([[A]]* [[AV]], i32 10)
    // CHECK: invoke void @_ZN5test11A3barEv([[A]]* [[AV]])
    // CHECK: call [[A]]* @_ZN5test11AD1Ev([[A]]* [[AV]])
    // CHECK: ret void
    A a = 10;
    a.bar();
  }

  // CHECK: define linkonce_odr [[A]]* @_ZN5test11AC1Ei([[A]]*
  // CHECK:   [[RET:%.*]] = alloca [[A]]*, align 4
  // CHECK:   [[THIS:%.*]] = alloca [[A]]*, align 4
  // CHECK:   store [[A]]* {{.*}}, [[A]]** [[THIS]]
  // CHECK:   [[THIS1:%.*]] = load [[A]]** [[THIS]]
  // CHECK:   store [[A]]* [[THIS1]], [[A]]** [[RET]]
  // CHECK:   call [[A]]* @_ZN5test11AC2Ei(
  // CHECK:   [[THIS2:%.*]] = load [[A]]** [[RET]]
  // CHECK:   ret [[A]]* [[THIS2]]

  // CHECK: define linkonce_odr [[A]]* @_ZN5test11AD1Ev([[A]]*
  // CHECK:   [[RET:%.*]] = alloca [[A]]*, align 4
  // CHECK:   [[THIS:%.*]] = alloca [[A]]*, align 4
  // CHECK:   store [[A]]* {{.*}}, [[A]]** [[THIS]]
  // CHECK:   [[THIS1:%.*]] = load [[A]]** [[THIS]]
  // CHECK:   store [[A]]* [[THIS1]], [[A]]** [[RET]]
  // CHECK:   call [[A]]* @_ZN5test11AD2Ev(
  // CHECK:   [[THIS2:%.*]] = load [[A]]** [[RET]]
  // CHECK:   ret [[A]]* [[THIS2]]
}

// Awkward virtual cases.
namespace test2 {
  void foo();

  struct A {
    int x;

    A(int);
    virtual ~A() { foo(); }
  };

  struct B {
    int y;
    int z;

    B(int);
    virtual ~B() { foo(); }
  };

  struct C : A, virtual B {
    int q;

    C(int i) : A(i), B(i) { foo(); }
    ~C() { foo(); }
  };

  void test() {
    C c = 10;
  }

  // CHECK: define linkonce_odr [[C:%.*]]* @_ZTv0_n12_N5test21CD1Ev(
  // CHECK:   call [[C]]* @_ZN5test21CD1Ev(
  // CHECK:   ret [[C]]* undef

  // CHECK: define linkonce_odr void @_ZTv0_n12_N5test21CD0Ev(
  // CHECK:   call void @_ZN5test21CD0Ev(
  // CHECK:   ret void
}

// CHECK: @_GLOBAL__D_a()
// CHECK: call %class.bar* @_ZN3barD1Ev(%class.bar* @baz)
