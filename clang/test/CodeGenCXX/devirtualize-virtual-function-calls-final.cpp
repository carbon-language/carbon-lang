// RUN: %clang_cc1 -std=c++11 %s -emit-llvm -o - | FileCheck %s

namespace Test1 {
  struct A {
    virtual int f() final;
  };

  // CHECK: define i32 @_ZN5Test11fEPNS_1AE
  int f(A *a) {
    // CHECK: call i32 @_ZN5Test11A1fEv
    return a->f();
  }
}

namespace Test2 {
  struct A final {
    virtual int f();
  };

  // CHECK: define i32 @_ZN5Test21fEPNS_1AE
  int f(A *a) {
    // CHECK: call i32 @_ZN5Test21A1fEv
    return a->f();
  }
}

namespace Test3 {
  struct A {
    virtual int f();
  };

  struct B final : A { };

  // CHECK: define i32 @_ZN5Test31fEPNS_1BE
  int f(B *b) {
    // CHECK: call i32 @_ZN5Test31A1fEv
    return b->f();
  }

  // CHECK: define i32 @_ZN5Test31fERNS_1BE
  int f(B &b) {
    // CHECK: call i32 @_ZN5Test31A1fEv
    return b.f();
  }

  // CHECK: define i32 @_ZN5Test31fEPv
  int f(void *v) {
    // CHECK: call i32 @_ZN5Test31A1fEv
    return static_cast<B*>(v)->f();
  }
}

namespace Test4 {
  struct A {
    virtual void f();
  };

  struct B final : A {
    virtual void f();
  };

  // CHECK: define void @_ZN5Test41fEPNS_1BE
  void f(B* d) {
    // CHECK: call void @_ZN5Test41B1fEv
    static_cast<A*>(d)->f();
  }
}

namespace Test5 {
  struct A {
    virtual void f();
  };

  struct B : A {
    virtual void f();
  };

  struct C final : B {
  };

  // CHECK: define void @_ZN5Test51fEPNS_1CE
  void f(C* d) {
    // CHECK: call void @_ZN5Test51B1fEv
    static_cast<A*>(d)->f();
  }
}

namespace Test6 {
  struct A {
    virtual ~A();
  };

  struct B : public A {
    virtual ~B();
  };

  struct C {
    virtual ~C();
  };

  struct D final : public C, public B {
  };

  // CHECK: define void @_ZN5Test61fEPNS_1DE
  void f(D* d) {
    // CHECK: call void @_ZN5Test61DD1Ev
    static_cast<A*>(d)->~A();
  }
}

namespace Test7 {
  struct foo {
    virtual void g() {}
  };

  struct bar {
    virtual int f() { return 0; }
  };

  struct zed final : public foo, public bar {
    int z;
    virtual int f() {return z;}
  };

  // CHECK: define i32 @_ZN5Test71fEPNS_3zedE
  int f(zed *z) {
    // CHECK: alloca
    // CHECK-NEXT: store
    // CHECK-NEXT: load
    // CHECK-NEXT: bitcast
    // CHECK-NEXT: call {{.*}} @_ZN5Test73zed1fEv
    // CHECK-NEXT: ret
    return static_cast<bar*>(z)->f();
  }
}
