// RUN: clang-cc -emit-llvm %s -o %t

void *test(int i)
{
  return (void *)i;
}
