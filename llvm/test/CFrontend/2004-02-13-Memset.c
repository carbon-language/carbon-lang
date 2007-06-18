// RUN: %llvmgcc -xc %s -c -o - | llvm-dis | grep llvm.memset | wc -l | grep 3

void test(int* X, char *Y) {
  memset(X, 4, 1000);
  bzero(Y, 100);
}
