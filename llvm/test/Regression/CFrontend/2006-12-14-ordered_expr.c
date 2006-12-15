// RUN: %llvmgcc -O3 -S %s -o - | grep llvm.isunordered &&
// RUN: %llvmgcc -O3 -S %s -o - | grep xor

int test2(float X, float Y) {
  return !__builtin_isunordered(X, Y);
}

