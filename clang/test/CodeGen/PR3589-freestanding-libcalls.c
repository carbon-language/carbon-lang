// RUN: clang -emit-llvm %s -o - | grep 'declare i32 @printf' | count 1 &&
// RUN: clang -O2 -emit-llvm %s -o - | grep 'declare i32 @puts' | count 1 &&
// RUN: clang -ffreestanding -O2 -emit-llvm %s -o - | grep 'declare i32 @puts' | count 0

#include <stdio.h>

void f0() {
  printf("hello\n");
}
