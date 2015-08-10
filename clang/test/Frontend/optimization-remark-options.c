// RUN: %clang -O1 -fvectorize -target x86_64-unknown-unknown -Rpass-analysis=loop-vectorize -emit-llvm -S %s -o - 2>&1 | FileCheck %s

// CHECK: {{.*}}:9:11: remark: loop not vectorized: vectorization requires changes in the order of operations, however IEEE 754 floating-point operations are not commutative; allow commutativity by specifying '#pragma clang loop vectorize(enable)' before the loop or by providing the compiler option '-ffast-math'

double foo(int N) {
  double v = 0.0;

  for (int i = 0; i < N; i++)
    v = v + 1.0;

  return v;
}

// CHECK: {{.*}}:18:13: remark: loop not vectorized: cannot prove pointers refer to independent arrays in memory. The loop requires 9 runtime independence checks to vectorize the loop, but that would exceed the limit of 8 checks; avoid runtime pointer checking when you know the arrays will always be independent by specifying '#pragma clang loop vectorize(assume_safety)' before the loop or by specifying 'restrict' on the array arguments. Erroneous results will occur if these options are incorrectly applied!

void foo2(int *dw, int *uw, int *A, int *B, int *C, int *D, int N) {
  for (int i = 0; i < N; i++) {
    dw[i] = A[i] + B[i - 1] + C[i - 2] + D[i - 3];
    uw[i] = A[i] + B[i + 1] + C[i + 2] + D[i + 3];
  }
}
