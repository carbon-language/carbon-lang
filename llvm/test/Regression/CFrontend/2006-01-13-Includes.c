// RUN: %llvmgcc %s -g -S -o - | gccas | llvm-dis | grep "llvm/test/Regression/CFrontend"

// PR676

#include <stdio.h>

void test() {
  printf("Hello World\n");
}
