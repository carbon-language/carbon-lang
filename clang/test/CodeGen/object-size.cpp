// RUN: %clang_cc1 -triple x86_64-apple-darwin -emit-llvm -o - %s | FileCheck %s

// C++-specific tests for __builtin_object_size

int gi;

// CHECK-LABEL: define void @_Z5test1v()
void test1() {
  // Guaranteeing that our cast removal logic doesn't break more interesting
  // cases.
  struct A { int a; };
  struct B { int b; };
  struct C: public A, public B {};

  C c;

  // CHECK: store i32 8
  gi = __builtin_object_size(&c, 0);
  // CHECK: store i32 8
  gi = __builtin_object_size((A*)&c, 0);
  // CHECK: store i32 4
  gi = __builtin_object_size((B*)&c, 0);

  // CHECK: store i32 8
  gi = __builtin_object_size((char*)&c, 0);
  // CHECK: store i32 8
  gi = __builtin_object_size((char*)(A*)&c, 0);
  // CHECK: store i32 4
  gi = __builtin_object_size((char*)(B*)&c, 0);
}
