// RUN: %llvmgxx %s -S -o - | grep '%XX = global int 4'

struct S {
  int  A[2];
};

int XX = (int)&(((struct S*)0)->A[1]);

