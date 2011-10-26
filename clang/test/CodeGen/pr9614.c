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
// CHECK: %call = call i32 @foo()
// CHECK: ret i32 %call
// CHECK-NOT: define
