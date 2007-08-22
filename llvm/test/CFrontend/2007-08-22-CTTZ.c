// RUN: %llvmgcc -O2 -S -o - -emit-llvm %s | grep {llvm.cttz.i64} | count 1
// RUN: %llvmgcc -O2 -S -o - -emit-llvm %s | not grep {lshr}

int bork(unsigned long long x) {
  return __builtin_ctzll(x);
}
