// RUN: clang -emit-llvm %s

int test(void* i)
{
  return (int)i;
}
