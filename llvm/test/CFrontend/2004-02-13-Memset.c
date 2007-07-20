// RUN: %llvmgcc -xc %s -c -O1 -o - | llvm-dis | grep llvm.memset | \
// RUN:   wc -l | grep 3

void test(int* X, char *Y) {
  memset(X, 4, 1000);
  bzero(Y, 100);
}
