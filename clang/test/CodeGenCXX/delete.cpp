// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

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

  // CHECK: define linkonce_odr void @_ZN5test01AD1Ev
  // CHECK: define linkonce_odr void @_ZN5test01AdlEPv
}
