// RUN: %clang -S -emit-llvm -o - -O2 %s | FileCheck %s -check-prefix=O2
// RUN: %clang -S -emit-llvm -o - -O0 %s | FileCheck %s -check-prefix=O0

extern int bar(char *A, int n);

// O0-NOT: @llvm.lifetime.start
int foo (int n) {
  if (n) {
// O2: @llvm.lifetime.start
    char A[100];
    return bar(A, 1);
  } else {
// O2: @llvm.lifetime.start
    char A[100];
    return bar(A, 2);
  }
}
