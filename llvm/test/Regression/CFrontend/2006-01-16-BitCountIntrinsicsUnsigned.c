// RUN: %llvmgcc -S %s -o - | grep 'llvm.ctlz.i32(uint' &&
// RUN: %llvmgcc -S %s -o - | not grep 'llvm.ctlz.i32(int'

unsigned t2(unsigned X) {
  return __builtin_clz(X);
}
int t1(int X) {
  return __builtin_clz(X);
}
