// RUN: clang-cc -triple i386-unknown-unknown -emit-llvm %s -o %t &&
// RUN: grep "load i8\*\*\* %p.addr"  %t | count 1

// PR3800
void f(void **p)
{
    __asm__ volatile("" :"+m"(*p));
}
