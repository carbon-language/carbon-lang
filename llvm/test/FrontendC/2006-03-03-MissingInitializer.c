// RUN: %llvmgcc %s -S -o - | opt -std-compile-opts | \
// RUN:    llvm-dis | grep {@nate.*internal unnamed_addr global i32 0}

struct X { int *XX; int Y;};

void foo() {
  static int nate = 0;
  struct X bob = { &nate, 14 };
  bar(&bob);
}

