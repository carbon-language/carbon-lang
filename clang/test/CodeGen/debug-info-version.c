// RUN: %clang -g -S -emit-llvm -o - %s | FileCheck %s
// RUN: %clang -S -emit-llvm -o - %s | FileCheck %s --check-prefix=NO_DEBUG
int main (void) {
  return 0;
}

// CHECK: metadata !{i32 2, metadata !"Debug Info Version", i32 2}
// NO_DEBUG-NOT: metadata !"Debug Info Version"
