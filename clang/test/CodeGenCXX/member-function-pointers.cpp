// RUN: clang-cc %s -emit-llvm -o - -triple=x86_64-apple-darwin9 | FileCheck %s

struct A { int a; void f(); virtual void vf(); };
struct B { int b; virtual void g(); };
struct C : B, A { };

void (A::*pa)();
void (A::*volatile vpa)();
void (B::*pb)();
void (C::*pc)();

// CHECK: @pa2 = global %0 { i64 ptrtoint (void ()* @_ZN1A1fEv to i64), i64 0 }, align 8
void (A::*pa2)() = &A::f;

// CHECK: @pa3 = global %0 { i64 1, i64 0 }, align 8
void (A::*pa3)() = &A::vf;

// CHECK: @pc2 = global %0 { i64 ptrtoint (void ()* @_ZN1A1fEv to i64), i64 16 }, align 8
void (C::*pc2)() = &C::f;

// CHECK: @pc3 = global %0 { i64 1, i64 0 }, align 8
void (A::*pc3)() = &A::vf;

void f() {
  // CHECK: store i64 0, i64* getelementptr inbounds (%0* @pa, i32 0, i32 0)
  // CHECK: store i64 0, i64* getelementptr inbounds (%0* @pa, i32 0, i32 1)
  pa = 0;

  // CHECK: volatile store i64 0, i64* getelementptr inbounds (%0* @vpa, i32 0, i32 0)
  // CHECK: volatile store i64 0, i64* getelementptr inbounds (%0* @vpa, i32 0, i32 1)
  vpa = 0;

  // CHECK: store i64 {{.*}}, i64* getelementptr inbounds (%0* @pc, i32 0, i32 0)
  // CHECK: [[ADJ:%[a-zA-Z0-9\.]+]] = add i64 {{.*}}, 16
  // CHECK: store i64 [[ADJ]], i64* getelementptr inbounds (%0* @pc, i32 0, i32 1)
  pc = pa;

  // CHECK: store i64 {{.*}}, i64* getelementptr inbounds (%0* @pa, i32 0, i32 0)
  // CHECK: [[ADJ:%[a-zA-Z0-9\.]+]] = sub i64 {{.*}}, 16
  // CHECK: store i64 [[ADJ]], i64* getelementptr inbounds (%0* @pa, i32 0, i32 1)
  pa = static_cast<void (A::*)()>(pc);
}

void f2() {
  // CHECK: [[pa2ptr:%[a-zA-Z0-9\.]+]] = getelementptr inbounds %0* %pa2, i32 0, i32 0 
  // CHECK: store i64 ptrtoint (void ()* @_ZN1A1fEv to i64), i64* [[pa2ptr]]
  // CHECK: [[pa2adj:%[a-zA-Z0-9\.]+]] = getelementptr inbounds %0* %pa2, i32 0, i32 1
  // CHECK: store i64 0, i64* [[pa2adj]]
  void (A::*pa2)() = &A::f;
  
  // CHECK: [[pa3ptr:%[a-zA-Z0-9\.]+]] = getelementptr inbounds %0* %pa3, i32 0, i32 0 
  // CHECK: store i64 1, i64* [[pa3ptr]]
  // CHECK: [[pa3adj:%[a-zA-Z0-9\.]+]] = getelementptr inbounds %0* %pa3, i32 0, i32 1
  // CHECK: store i64 0, i64* [[pa3adj]]
  void (A::*pa3)() = &A::vf;
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
