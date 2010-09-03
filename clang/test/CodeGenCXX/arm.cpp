// RUN: %clang_cc1 %s -triple=thumbv7-apple-darwin3.0.0-iphoneos -fno-use-cxa-atexit -target-abi apcs-gnu -emit-llvm -o - -fexceptions | FileCheck %s

typedef typeof(sizeof(int)) size_t;

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

  // Tests at eof
}

namespace test3 {
  struct A {
    int x;
    ~A();
  };

  void a() {
    // CHECK: define void @_ZN5test31aEv()
    // CHECK: call noalias i8* @_Znam(i32 48)
    // CHECK: store i32 4
    // CHECK: store i32 10
    A *x = new A[10];
  }

  void b(int n) {
    // CHECK: define void @_ZN5test31bEi(
    // CHECK: [[N:%.*]] = load i32*
    // CHECK: @llvm.umul.with.overflow.i32(i32 [[N]], i32 4)
    // CHECK: @llvm.uadd.with.overflow.i32(i32 {{.*}}, i32 8)
    // CHECK: [[SZ:%.*]] = select
    // CHECK: call noalias i8* @_Znam(i32 [[SZ]])
    // CHECK: store i32 4
    // CHECK: store i32 [[N]]
    A *x = new A[n];
  }

  void c() {
    // CHECK: define void @_ZN5test31cEv()
    // CHECK: call  noalias i8* @_Znam(i32 808)
    // CHECK: store i32 4
    // CHECK: store i32 200
    A (*x)[20] = new A[10][20];
  }

  void d(int n) {
    // CHECK: define void @_ZN5test31dEi(
    // CHECK: [[N:%.*]] = load i32*
    // CHECK: [[NE:%.*]] = mul i32 [[N]], 20
    // CHECK: @llvm.umul.with.overflow.i32(i32 [[N]], i32 80)
    // CHECK: @llvm.uadd.with.overflow.i32(i32 {{.*}}, i32 8)
    // CHECK: [[SZ:%.*]] = select
    // CHECK: call noalias i8* @_Znam(i32 [[SZ]])
    // CHECK: store i32 4
    // CHECK: store i32 [[NE]]
    A (*x)[20] = new A[n][20];
  }

  void e(A *x) {
    // CHECK: define void @_ZN5test31eEPNS_1AE(
    // CHECK: icmp eq {{.*}}, null
    // CHECK: getelementptr {{.*}}, i64 -8
    // CHECK: getelementptr {{.*}}, i64 4
    // CHECK: bitcast {{.*}} to i32*
    // CHECK: load
    // CHECK: invoke {{.*}} @_ZN5test31AD1Ev
    // CHECK: call void @_ZdaPv
    delete [] x;
  }

  void f(A (*x)[20]) {
    // CHECK: define void @_ZN5test31fEPA20_NS_1AE(
    // CHECK: icmp eq {{.*}}, null
    // CHECK: getelementptr {{.*}}, i64 -8
    // CHECK: getelementptr {{.*}}, i64 4
    // CHECK: bitcast {{.*}} to i32*
    // CHECK: load
    // CHECK: invoke {{.*}} @_ZN5test31AD1Ev
    // CHECK: call void @_ZdaPv
    delete [] x;
  }
}

namespace test4 {
  struct A {
    int x;
    void operator delete[](void *, size_t sz);
  };

  void a() {
    // CHECK: define void @_ZN5test41aEv()
    // CHECK: call noalias i8* @_Znam(i32 48)
    // CHECK: store i32 4
    // CHECK: store i32 10
    A *x = new A[10];
  }

  void b(int n) {
    // CHECK: define void @_ZN5test41bEi(
    // CHECK: [[N:%.*]] = load i32*
    // CHECK: @llvm.umul.with.overflow.i32(i32 [[N]], i32 4)
    // CHECK: @llvm.uadd.with.overflow.i32(i32 {{.*}}, i32 8)
    // CHECK: [[SZ:%.*]] = select
    // CHECK: call noalias i8* @_Znam(i32 [[SZ]])
    // CHECK: store i32 4
    // CHECK: store i32 [[N]]
    A *x = new A[n];
  }

  void c() {
    // CHECK: define void @_ZN5test41cEv()
    // CHECK: call  noalias i8* @_Znam(i32 808)
    // CHECK: store i32 4
    // CHECK: store i32 200
    A (*x)[20] = new A[10][20];
  }

  void d(int n) {
    // CHECK: define void @_ZN5test41dEi(
    // CHECK: [[N:%.*]] = load i32*
    // CHECK: [[NE:%.*]] = mul i32 [[N]], 20
    // CHECK: @llvm.umul.with.overflow.i32(i32 [[N]], i32 80)
    // CHECK: @llvm.uadd.with.overflow.i32(i32 {{.*}}, i32 8)
    // CHECK: [[SZ:%.*]] = select
    // CHECK: call noalias i8* @_Znam(i32 [[SZ]])
    // CHECK: store i32 4
    // CHECK: store i32 [[NE]]
    A (*x)[20] = new A[n][20];
  }

  void e(A *x) {
    // CHECK: define void @_ZN5test41eEPNS_1AE(
    // CHECK: [[ALLOC:%.*]] = getelementptr inbounds {{.*}}, i64 -8
    // CHECK: getelementptr inbounds {{.*}}, i64 4
    // CHECK: bitcast
    // CHECK: [[T0:%.*]] = load i32*
    // CHECK: [[T1:%.*]] = mul i32 4, [[T0]]
    // CHECK: [[T2:%.*]] = add i32 [[T1]], 8
    // CHECK: call void @_ZN5test41AdaEPvm(i8* [[ALLOC]], i32 [[T2]])
    delete [] x;
  }

  void f(A (*x)[20]) {
    // CHECK: define void @_ZN5test41fEPA20_NS_1AE(
    // CHECK: [[ALLOC:%.*]] = getelementptr inbounds {{.*}}, i64 -8
    // CHECK: getelementptr inbounds {{.*}}, i64 4
    // CHECK: bitcast
    // CHECK: [[T0:%.*]] = load i32*
    // CHECK: [[T1:%.*]] = mul i32 4, [[T0]]
    // CHECK: [[T2:%.*]] = add i32 [[T1]], 8
    // CHECK: call void @_ZN5test41AdaEPvm(i8* [[ALLOC]], i32 [[T2]])
    delete [] x;
  }
}

// <rdar://problem/8386802>: don't crash
namespace test5 {
  struct A {
    ~A();
  };

  // CHECK: define void @_ZN5test54testEPNS_1AE
  void test(A *a) {
    // CHECK:      [[PTR:%.*]] = alloca [[A:%.*]]*, align 4
    // CHECK-NEXT: store [[A]]* {{.*}}, [[A]]** [[PTR]], align 4
    // CHECK-NEXT: [[TMP:%.*]] = load [[A]]** [[PTR]], align 4
    // CHECK-NEXT: call [[A]]* @_ZN5test51AD1Ev([[A]]* [[TMP]])
    // CHECK-NEXT: ret void
    a->~A();
  }
}

  // CHECK: define linkonce_odr [[C:%.*]]* @_ZTv0_n12_N5test21CD1Ev(
  // CHECK:   call [[C]]* @_ZN5test21CD1Ev(
  // CHECK:   ret [[C]]* undef

  // CHECK: define linkonce_odr void @_ZTv0_n12_N5test21CD0Ev(
  // CHECK:   call void @_ZN5test21CD0Ev(
  // CHECK:   ret void

// CHECK: @_GLOBAL__D_a()
// CHECK: call %class.bar* @_ZN3barD1Ev(%class.bar* @baz)
