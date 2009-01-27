// Check that object files compiled with -mdynamic-no-pic can be
// linked.
// 
// RUN: xcc -ccc-clang -m32 -mdynamic-no-pic %s -c -o %t.o &&
// RUN: xcc -ccc-clang -m32 %t.o -o %t &&
// RUN: %t | grep "Hello, World" &&
// RUN: xcc -ccc-clang -m64 -mdynamic-no-pic %s -c -o %t.o &&
// RUN: xcc -ccc-clang -m64 %t.o -o %t &&
// RUN: %t | grep "Hello, World" &&
// RUN: true

#include <stdio.h>

int main(int argc, char **argv) {
  fprintf(stdout, "Hello, World");
  return 0;
}
