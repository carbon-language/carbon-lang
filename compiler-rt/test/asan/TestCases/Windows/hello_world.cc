// RUN: %clangxx_asan -O0 %s -Fe%t
// RUN: %run %t | FileCheck %s

#include <stdio.h>

int main() {
  printf("Hello, world!\n");
// CHECK: Hello, world!
}
