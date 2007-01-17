// RUN: %llvmgxx %s -S -o - | not grep 'llvm.global_ctor'

struct S {
  int  A[2];
};

int XX = (int)&(((struct S*)0)->A[1]);

