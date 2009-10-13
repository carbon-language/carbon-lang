// RUN: %llvmgcc %s -g -S -o - | grep {test/FrontendC}
// PR676

#include <stdio.h>

void test() {
  printf("Hello World\n");
}
