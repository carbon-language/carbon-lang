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

void fold_for(int a, int b) {
  // CHECK: define {{.*}} @fold_for(
  // CHECK-NOT: = phi
  // CHECK: }
  for (int i = 0; a && i < b; ++i) foo();
  for (int i = 0; a || i < b; ++i) bar();
}
