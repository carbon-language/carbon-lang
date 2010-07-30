// RUN: %llvmgcc_only -std=gnu99 %s -S |& grep {warning: alignment for}

int foo(int a)
{
  int var[a] __attribute__((__aligned__(32)));
  return 4;
}
