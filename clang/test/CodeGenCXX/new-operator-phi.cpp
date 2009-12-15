// RUN: %clang_cc1 -emit-llvm-only -verify %s
// PR5454
#include <stddef.h>

class X {static void * operator new(size_t size) throw(); X(int); };
int a(), b();
void b(int x)
{
  new X(x ? a() : b());
}

