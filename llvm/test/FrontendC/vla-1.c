// RUN: %llvmgcc_only -std=gnu99 %s -S |& grep {warning: alignment for}
// ppc does not support this feature, and gets a fatal error at runtime.
// XFAIL: powerpc

int foo(int a)
{
  int var[a] __attribute__((__aligned__(32)));
  return 4;
}
