// RUN: clang -emit-llvm -o %t %s

#include <stdio.h>

void foo(id a) {
  @synchronized(a) {
    printf("Swimming? No.");
    return;
  }
}

