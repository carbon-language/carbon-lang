// RUN: %llvmgcc %s -S -o - | gccas | llvm-dis | grep llvm.stacksave
// XFAIL: *

// PR691

void test(int N) {
  int i;
  for (i = 0; i < N; ++i) {
    int VLA[i];
    external(VLA);
  }
}
