// RUN: true
// %llvmgcc -std=gnu99 %s -S |& grep {error: "is greater than the stack alignment" } 

int foo(int a)
{
  int var[a] __attribute__((__aligned__(32)));
  return 4;
}
