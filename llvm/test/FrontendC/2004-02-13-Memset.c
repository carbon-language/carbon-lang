// RUN: %llvmgcc -xc %s -c -o - | llvm-dis | grep llvm.memset | count 3

#include <memory.h>

void test(int* X, char *Y) {
  memset(X, 4, 1000);
  bzero(Y, 100);
}
