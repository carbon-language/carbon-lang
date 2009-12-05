// PR691
// RUN: %llvmgcc %s -S -o - | opt -std-compile-opts | \
// RUN:    llvm-dis | grep llvm.stacksave

void test(int N) {
  int i;
  for (i = 0; i < N; ++i) {
    int VLA[i];
    external(VLA);
  }
}
