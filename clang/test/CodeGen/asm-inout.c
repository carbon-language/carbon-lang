// RUN: clang-cc -triple i386-unknown-unknown -emit-llvm %s -o %t
// RUN: grep "load i8\*\*\* %p.addr"  %t | count 1
// XFAIL: *

// PR3800
void f(void **p)
{
    __asm__ volatile("" :"+m"(*p));
}

#if 0
// FIXME: Once this works again, we must verify that the code below behaves as expected
// See PR4677.
void f() {
  unsigned _data = 42;
  __asm__("bswap   %0":"+r"(_data));
}
#endif
