// RUN: %llvmgcc -std=gnu99 %s -S -o - | grep ".*alloca.*align 16"

extern void bar(int[]);

void foo(int a)
{
  int var[a] __attribute__((__aligned__(16)));
  bar(var);
  return;
}
