// RUN: clang -emit-llvm -o %t %s &&
// RUN: grep -e "global_ctors.*@A" %t &&
// RUN: grep -e "global_dtors.*@B" %t

#include <stdio.h>

void A() __attribute__((constructor));
void B() __attribute__((destructor));

void A() {
  printf("A\n");
}

void B() {
  printf("B\n");
}

int main() {
  return 0;
}
