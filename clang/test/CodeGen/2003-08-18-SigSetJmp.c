// RUN: %clang -S -emit-llvm %s  -o /dev/null
// XFAIL: mingw,win32

#include <setjmp.h>

sigjmp_buf B;
int foo() {
  sigsetjmp(B, 1);
  bar();
}
