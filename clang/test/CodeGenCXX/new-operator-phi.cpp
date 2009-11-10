// RUN: clang-cc -emit-llvm-only -verify %s
// PR5454

class X {static void * operator new(unsigned long size) throw(); X(int); };
int a(), b();
void b(int x)
{
  new X(x ? a() : b());
}

