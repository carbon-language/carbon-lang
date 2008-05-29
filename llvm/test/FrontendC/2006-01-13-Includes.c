// RUN: %llvmgcc %s -g -S -o - | llvm-as | opt -std-compile-opts | \
// RUN:   llvm-dis | grep {test/CFrontend}
// PR676

#include <stdio.h>

void test() {
  printf("Hello World\n");
}
