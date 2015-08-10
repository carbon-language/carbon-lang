// RUN: %clang -O1 -fvectorize -Rpass-analysis=loop-vectorize -emit-llvm -S %s -o - 2>&1 | FileCheck %s

// CHECK: {{.*}}:9:11: remark: loop not vectorized: vectorization requires changes in the order of operations, however IEEE 754 floating-point operations are not commutative; allow commutativity by specifying ‘#pragma clang loop vectorize(enable)’ before the loop or by providing the compiler option ‘-ffast-math’

double foo(int N) {
  double v = 0.0;

  for (int i = 0; i < N; i++)
    v = v + 1.0;

  return v;
}
