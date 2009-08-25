// RUN: %llvmgcc -S %s -o - | llvm-as -o /dev/null


int foo(int *A, unsigned X) {
  return A[X];
}
