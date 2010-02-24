// RUN: %clang_cc1 %s -emit-llvm -o - -verify | FileCheck %s

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

void f0() {}
// CHECK: define void @f0()

void f1();
void f2(void) {
// CHECK: call void @f1()
  f1(1, 2, 3);
}
// CHECK: define void @f1()
void f1() {}

// CHECK: define {{.*}} @f3{{\(\)|\(.*sret.*\)}}
struct foo { int X, Y, Z; } f3() {
  while (1) {}
}

// PR4423 - This shouldn't crash in codegen
void f4() {}
void f5() { f4(42); } //expected-warning {{too many arguments}}

// Qualifiers on parameter types shouldn't make a difference.
static void f6(const float f, const float g) {
}
void f7(float f, float g) {
  f6(f, g);
// CHECK: define void @f7(float{{.*}}, float{{.*}})
// CHECK: call void @f6(float{{.*}}, float{{.*}})
}
