// RUN: %llvmgcc %s -g -S -o - | gccas | llvm-dis | grep "test/Regression/CFrontend"
// XFAIL: llvmgcc4
// PR676

#include <stdio.h>

void test() {
  printf("Hello World\n");
}
