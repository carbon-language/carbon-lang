// RUN: %clang %s -O0 -emit-llvm -S -o - | FileCheck %s

void foo();
void bar();

void fold_if(int a, int b) {
  // CHECK: define {{.*}} @fold_if(
  // CHECK-NOT: = phi
  // CHECK: }
  if (a && b)
    foo();
  else
    bar();
}
