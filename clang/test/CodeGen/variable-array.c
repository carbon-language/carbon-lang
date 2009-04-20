// RUN: clang-cc -emit-llvm < %s | grep puts

int a(int x)
{
  int (*y)[x];
  return sizeof(*(puts("asdf"),y));
}
