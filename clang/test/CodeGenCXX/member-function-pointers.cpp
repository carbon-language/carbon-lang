// RUN: %clang_cc1 %s -emit-llvm -o - -triple=x86_64-apple-darwin9 | FileCheck %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=i386-apple-darwin9 | FileCheck -check-prefix LP32 %s
// RUN: %clang_cc1 %s -emit-llvm -o - -triple=armv7-unknown-unknown | FileCheck -check-prefix ARM %s

struct A { int a; void f(); virtual void vf1(); virtual void vf2(); };
struct B { int b; virtual void g(); };
struct C : B, A { };

void (A::*pa)();
void (A::*volatile vpa)();
void (B::*pb)();
void (C::*pc)();

// CHECK: @pa2 = global { i64, i64 } { i64 ptrtoint (void (%struct.A*)* @_ZN1A1fEv to i64), i64 0 }, align 8
void (A::*pa2)() = &A::f;

// CHECK:      @pa3 = global { i64, i64 } { i64 1, i64 0 }, align 8
// CHECK-LP32: @pa3 = global { i32, i32 } { i32 1, i32 0 }, align 4
void (A::*pa3)() = &A::vf1;

// CHECK:      @pa4 = global { i64, i64 } { i64 9, i64 0 }, align 8
// CHECK-LP32: @pa4 = global { i32, i32 } { i32 5, i32 0 }, align 4
void (A::*pa4)() = &A::vf2;

// CHECK: @pc2 = global { i64, i64 } { i64 ptrtoint (void (%struct.A*)* @_ZN1A1fEv to i64), i64 16 }, align 8
void (C::*pc2)() = &C::f;

// CHECK: @pc3 = global { i64, i64 } { i64 1, i64 0 }, align 8
void (A::*pc3)() = &A::vf1;

void f() {
  // CHECK: store { i64, i64 } zeroinitializer, { i64, i64 }* @pa
  pa = 0;

  // Is this okay?  What are LLVM's volatile semantics for structs?
  // CHECK: store volatile { i64, i64 } zeroinitializer, { i64, i64 }* @vpa
  vpa = 0;

  // CHECK: [[TMP:%.*]] = load { i64, i64 }* @pa, align 8
  // CHECK: [[TMPADJ:%.*]] = extractvalue { i64, i64 } [[TMP]], 1
  // CHECK: [[ADJ:%.*]] = add nsw i64 [[TMPADJ]], 16
  // CHECK: [[RES:%.*]] = insertvalue { i64, i64 } [[TMP]], i64 [[ADJ]], 1
  // CHECK: store { i64, i64 } [[RES]], { i64, i64 }* @pc, align 8
  pc = pa;

  // CHECK: [[TMP:%.*]] = load { i64, i64 }* @pc, align 8
  // CHECK: [[TMPADJ:%.*]] = extractvalue { i64, i64 } [[TMP]], 1
  // CHECK: [[ADJ:%.*]] = sub nsw i64 [[TMPADJ]], 16
  // CHECK: [[RES:%.*]] = insertvalue { i64, i64 } [[TMP]], i64 [[ADJ]], 1
  // CHECK: store { i64, i64 } [[RES]], { i64, i64 }* @pa, align 8
  pa = static_cast<void (A::*)()>(pc);
}

void f2() {
  // CHECK:      store { i64, i64 } { i64 ptrtoint (void (%struct.A*)* @_ZN1A1fEv to i64), i64 0 }
  void (A::*pa2)() = &A::f;
  
  // CHECK:      store { i64, i64 } { i64 1, i64 0 }
  // CHECK-LP32: store { i32, i32 } { i32 1, i32 0 }
  void (A::*pa3)() = &A::vf1;
  
  // CHECK:      store { i64, i64 } { i64 9, i64 0 }
  // CHECK-LP32: store { i32, i32 } { i32 5, i32 0 }
  void (A::*pa4)() = &A::vf2;
}

void f3(A *a, A &ar) {
  (a->*pa)();
  (ar.*pa)();
}

bool f4() {
  return pa;
}

// PR5177
namespace PR5177 {
  struct A {
   bool foo(int*) const;
  } a;

  struct B1 {
   bool (A::*pmf)(int*) const;
   const A* pa;

   B1() : pmf(&A::foo), pa(&a) {}
   bool operator()() const { return (pa->*pmf)(new int); }
  };

  void bar(B1 b2) { while (b2()) ; }
}

// PR5138
namespace PR5138 {
  struct foo {
      virtual void bar(foo *);
  };

  extern "C" {
    void baz(foo *);
  }
  
  void (foo::*ptr1)(void *) = (void (foo::*)(void *))&foo::bar;
  void (*ptr2)(void *) = (void (*)(void *))&baz;

  void (foo::*ptr3)(void) = (void (foo::*)(void))&foo::bar;
}

// PR5593
namespace PR5593 {
  struct A { };
  
  bool f(void (A::*f)()) {
    return f && f;
  }
}

namespace PR5718 {
  struct A { };
  
  bool f(void (A::*f)(), void (A::*g)()) {
    return f == g;
  }
}

namespace BoolMemberPointer {
  struct A { };
  
  bool f(void (A::*f)()) {
    return !f;
  }

  bool g(void (A::*f)()) {
    if (!!f)
      return true;
    return false;
  }
}

// PR5940
namespace PR5940 {
  class foo {
  public:
    virtual void baz(void);
  };

  void foo::baz(void) {
       void (foo::*ptr)(void) = &foo::baz;
  }
}

namespace MemberPointerImpCast {
  struct A {
    int x;
  };
  struct B : public A {
  };
  void f(B* obj, void (A::*method)()) {
    (obj->*method)();
  }
}

// PR6258
namespace PR6258 {

  struct A {
    void f(bool);
  };

  void (A::*pf)(bool) = &A::f;

  void f() {
    void (A::*pf)(bool) = &A::f;
  }
}

// PR7027 
namespace PR7027 {
  struct X { void test( ); };
  void testX() { &X::test; }
}

namespace test7 {
  struct A { void foo(); virtual void vfoo(); };
  struct B { void foo(); virtual void vfoo(); };
  struct C : A, B { void foo(); virtual void vfoo(); };

  // CHECK-ARM: @_ZN5test74ptr0E = global {{.*}} { i32 ptrtoint ({{.*}}* @_ZN5test71A3fooEv to i32), i32 0 }
  // CHECK-ARM: @_ZN5test74ptr1E = global {{.*}} { i32 ptrtoint ({{.*}}* @_ZN5test71B3fooEv to i32), i32 8 }
  // CHECK-ARM: @_ZN5test74ptr2E = global {{.*}} { i32 ptrtoint ({{.*}}* @_ZN5test71C3fooEv to i32), i32 0 }
  // CHECK-ARM: @_ZN5test74ptr3E = global {{.*}} { i32 0, i32 1 }
  // CHECK-ARM: @_ZN5test74ptr4E = global {{.*}} { i32 0, i32 9 }
  // CHECK-ARM: @_ZN5test74ptr5E = global {{.*}} { i32 0, i32 1 }
  void (C::*ptr0)() = &A::foo;
  void (C::*ptr1)() = &B::foo;
  void (C::*ptr2)() = &C::foo;
  void (C::*ptr3)() = &A::vfoo;
  void (C::*ptr4)() = &B::vfoo;
  void (C::*ptr5)() = &C::vfoo;
}

namespace test8 {
  struct X { };
  typedef int (X::*pmf)(int);
  
  // CHECK: {{define.*_ZN5test81fEv}}
  pmf f() {
    // CHECK: {{ret.*zeroinitializer}}
    return pmf();
  }
}

namespace test9 {
  struct A {
    void foo();
  };
  struct B : A {
    void foo();
  };

  typedef void (A::*fooptr)();

  struct S {
    fooptr p;
  };

  // CHECK:    define void @_ZN5test94testEv(
  // CHECK:      alloca i32
  // CHECK-NEXT: ret void
  void test() {
    int x;
    static S array[] = { (fooptr) &B::foo };
  }
}
