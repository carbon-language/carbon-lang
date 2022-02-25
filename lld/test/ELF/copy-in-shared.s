// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/copy-in-shared.s -o %t1.o
// RUN: ld.lld -shared %t1.o -o %t1.so
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t2.o
// RUN: not ld.lld %t2.o %t1.so -o /dev/null -shared 2>&1 | FileCheck %s

// CHECK: error: relocation R_X86_64_64 cannot be used against symbol 'foo'; recompile with -fPIC
// CHECK: >>> defined in {{.*}}.so
// CHECK: >>> referenced by {{.*}}.o:(.text+0x0)

.quad foo
