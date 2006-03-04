// RUN: %llvmgcc %s -S -o - | gccas | llvm-dis | grep nate | grep 'global int 0'

struct X { int *XX; int Y;};

void foo() {
  static int nate = 0;
  struct X bob = { &nate, 14 };
  bar(&bob);
}

