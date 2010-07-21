// RUN: not %llvmgcc -std=gnu99 %s -S |& grep "error: alignment for"

int foo(int a)
{
  int var[a] __attribute__((__aligned__(32)));
  return 4;
}
