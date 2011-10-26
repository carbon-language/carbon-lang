// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

extern int foo_alias (void) __asm ("foo");
inline int foo (void) {
  return foo_alias ();
}
int f(void) {
  return foo();
}

// CHECK-NOT: define
// CHECK: define i32 @f()
// CHECK: call i32 @foo()
// CHECK-NEXT: ret i32
// CHECK-NOT: define
