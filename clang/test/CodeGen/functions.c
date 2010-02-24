// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s

int g();

int foo(int i) {
  return g(i);
}

int g(int i) {
  return g(i);
}

// rdar://6110827
typedef void T(void);
void test3(T f) {
  f();
}

int a(int);
int a() {return 1;}

// RUN: grep 'define void @f0()' %t
void f0() {}

void f1();
// RUN: grep 'call void @f1()' %t
void f2(void) {
  f1(1, 2, 3);
}
// RUN: grep 'define void @f1()' %t
void f1() {}

// RUN: grep 'define .* @f3' %t | not grep -F '...'
struct foo { int X, Y, Z; } f3() {
  while (1) {}
}

// PR4423 - This shouldn't crash in codegen
void f4() {}
void f5() { f4(42); }

// Qualifiers on parameter types shouldn't make a difference.
static void f6(const float f, const float g) {
}
void f7(float f, float g) {
  f6(f, g);
// CHECK: define void @f7(float{{.*}}, float{{.*}})
// CHECK: call void @f6(float{{.*}}, float{{.*}})
}
