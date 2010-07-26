// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

int x;
int y(void);
void foo();
void FUNC() {
// CHECK: [[call:%.*]] = call i32 @y
  if (__builtin_expect (x, y()))
    foo ();
}

