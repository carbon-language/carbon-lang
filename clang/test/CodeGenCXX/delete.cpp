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

  // CHECK: define linkonce_odr void @_ZN5test01AD1Ev(%"struct.test0::A"* %this) unnamed_addr
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
    // CHECK:      icmp eq [10 x [20 x [[A:%.*]]]]* [[PTR:%.*]], null
    // CHECK-NEXT: br i1

    // CHECK:      [[BEGIN:%.*]] = getelementptr inbounds [10 x [20 x [[A]]]]* [[PTR]], i32 0, i32 0, i32 0
    // CHECK-NEXT: [[T0:%.*]] = bitcast [[A]]* [[BEGIN]] to i8*
    // CHECK-NEXT: [[ALLOC:%.*]] = getelementptr inbounds i8* [[T0]], i64 -8
    // CHECK-NEXT: [[T1:%.*]] = bitcast i8* [[ALLOC]] to i64*
    // CHECK-NEXT: [[COUNT:%.*]] = load i64* [[T1]]
    // CHECK-NEXT: [[ISZERO:%.*]] = icmp eq i64 [[COUNT]], 0
    // CHECK-NEXT: br i1 [[ISZERO]],
    // CHECK:      [[END:%.*]] = getelementptr inbounds [[A]]* [[BEGIN]], i64 [[COUNT]]
    // CHECK-NEXT: br label
    // CHECK:      [[PAST:%.*]] = phi [[A]]* [ [[END]], {{%.*}} ], [ [[CUR:%.*]], {{%.*}} ]
    // CHECK-NEXT: [[CUR:%.*]] = getelementptr inbounds [[A]]* [[PAST]], i64 -1
    // CHECK-NEXT: call void @_ZN5test11AD1Ev([[A]]* [[CUR]])
    // CHECK-NEXT: [[ISDONE:%.*]] = icmp eq [[A]]* [[CUR]], [[BEGIN]]
    // CHECK-NEXT: br i1 [[ISDONE]]
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

namespace test4 {
  // PR10341: ::delete with a virtual destructor
  struct X {
    virtual ~X();
    void operator delete (void *);
  };

  // CHECK: define void @_ZN5test421global_delete_virtualEPNS_1XE
  void global_delete_virtual(X *xp) {
    // CHECK: [[VTABLE:%.*]] = load void ([[X:%.*]])***
    // CHECK-NEXT: [[VFN:%.*]] = getelementptr inbounds void ([[X]])** [[VTABLE]], i64 0
    // CHECK-NEXT: [[VFNPTR:%.*]] = load void ([[X]])** [[VFN]]
    // CHECK-NEXT: call void [[VFNPTR]]([[X]] [[OBJ:%.*]])
    // CHECK-NEXT: [[OBJVOID:%.*]] = bitcast [[X]] [[OBJ]] to i8*
    // CHECK-NEXT: call void @_ZdlPv(i8* [[OBJVOID]]) nounwind
    ::delete xp;
  }
}
