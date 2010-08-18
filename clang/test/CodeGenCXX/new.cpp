// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -emit-llvm -o - | FileCheck %s
#include <stddef.h>

void t1() {
  int* a = new int;
}

// Placement.
void* operator new(size_t, void*) throw();

void t2(int* a) {
  int* b = new (a) int;
}

struct S {
  int a;
};

// POD types.
void t3() {
  int *a = new int(10);
  _Complex int* b = new _Complex int(10i);
  
  S s;
  s.a = 10;
  S *sp = new S(s);
}

// Non-POD
struct T {
  T();
  int a;
};

void t4() {
  // CHECK: call void @_ZN1TC1Ev
  T *t = new T;
}

struct T2 {
  int a;
  T2(int, int);
};

void t5() { 
  // CHECK: call void @_ZN2T2C1Eii
  T2 *t2 = new T2(10, 10);
}

int *t6() {
  // Null check.
  return new (0) int(10);
}

void t7() {
  new int();
}

struct U {
  ~U();
};
  
void t8(int n) {
  new int[10];
  new int[n];
  
  // Non-POD
  new T[10];
  new T[n];
  
  // Cookie required
  new U[10];
  new U[n];
}

// noalias
// CHECK: declare noalias i8* @_Znam
void *operator new[](size_t);

void t9() {
  bool b;

  new bool(true);  
  new (&b) bool(true);
}

struct A {
  void* operator new(__typeof(sizeof(int)), int, float, ...);
  A();
};

A* t10() {
   // CHECK: @_ZN1AnwEmifz
  return new(1, 2, 3.45, 100) A;
}

// CHECK: define void @_Z3t11i
struct B { int a; };
struct Bmemptr { int Bmemptr::* memptr; int a; };

void t11(int n) {
  // CHECK: call noalias i8* @_Znwm
  // CHECK: call void @llvm.memset.p0i8.i64(
  B* b = new B();

  // CHECK: call noalias i8* @_Znam
  // CHECK: {{call void.*llvm.memset.p0i8.i64.*i8 0, i64 %}}
  B *b2 = new B[n]();

  // CHECK: call noalias i8* @_Znam
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
  // CHECK: br
  Bmemptr *b_memptr = new Bmemptr[n]();
  
  // CHECK: ret void
}

struct Empty { };

// We don't need to initialize an empty class.
// CHECK: define void @_Z3t12v
void t12() {
  // CHECK: call noalias i8* @_Znam
  // CHECK-NOT: br
  (void)new Empty[10];

  // CHECK: call noalias i8* @_Znam
  // CHECK-NOT: br
  (void)new Empty[10]();

  // CHECK: ret void
}

// Zero-initialization
// CHECK: define void @_Z3t13i
void t13(int n) {
  // CHECK: call noalias i8* @_Znwm
  // CHECK: store i32 0, i32*
  (void)new int();

  // CHECK: call noalias i8* @_Znam
  // CHECK: {{call void.*llvm.memset.p0i8.i64.*i8 0, i64 %}}
  (void)new int[n]();

  // CHECK-NEXT: ret void
}
