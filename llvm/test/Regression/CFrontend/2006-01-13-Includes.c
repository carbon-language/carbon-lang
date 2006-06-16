// RUN: %llvmgcc %s -g -S -o - | gccas | llvm-dis | grep "test/Regression/CFrontend"

#include <stdio.h>

void test() {
  printf("Hello World\n");
}
