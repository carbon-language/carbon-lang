// RUN: %clang_cc1  %s -emit-llvm -o - | grep llvm.memset | count 3

void *memset(void*, int, unsigned long);
void bzero(void*, unsigned long);

void test(int* X, char *Y) {
  // CHECK: call i8* llvm.memset
  memset(X, 4, 1000);
  // CHECK: call void bzero
  bzero(Y, 100);
}
