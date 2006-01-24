// RUN: %llvmgcc %s -S -o - | gccas | llvm-dis | grep foo[12345] | wc -l | grep 5
// XFAIL: *

__asm__ ("foo1");
__asm__ ("foo2");
__asm__ ("foo3");
__asm__ ("foo4");
__asm__ ("foo5");
