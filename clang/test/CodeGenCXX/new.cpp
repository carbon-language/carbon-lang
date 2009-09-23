// RUN: clang-cc %s -emit-llvm -o - | FileCheck %s

void t1() {
  int* a = new int;
}

// Placement.
void* operator new(unsigned long, void*) throw();

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
