// RUN: %clang_cc1 %s -emit-llvm -O0 -o - | FileCheck %s
// pr6552

// XFAIL: *
// XTARGET: arm

extern void bar(unsigned int ip);

// CHECK: mov r0, r12
void foo(void)
{
  register unsigned int ip __asm ("ip");
  bar(ip);
}

