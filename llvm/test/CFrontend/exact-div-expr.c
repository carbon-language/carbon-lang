// RUN: %llvmgcc -S %s -o - -O | grep ashr
// RUN: %llvmgcc -S %s -o - -O | not grep sdiv

long long test(int *A, int *B) {
  return A-B;
}
