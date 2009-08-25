// RUN: %llvmgcc -S %s -o - | llvm-as -o /dev/null

#include <stdio.h>

void  test() {
  fprintf(stderr, "testing\n");
}
