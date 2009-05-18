// RUN: %llvmgcc -S %s -O0 -o - -mllvm -disable-llvm-optzns | grep bar
// Check that the gcc inliner is turned off.

#include <stdio.h>
static __inline__ __attribute__ ((always_inline))
     int bar (int x)
{
  return 4;
}

void
foo ()
{
  long long b = 1;
  int Y = bar (4);
  printf ("%d\n", Y);
}
