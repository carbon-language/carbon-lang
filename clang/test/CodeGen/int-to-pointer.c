// RUN: clang -emit-llvm %s

void *test(int i)
{
  return (void *)i;
}
