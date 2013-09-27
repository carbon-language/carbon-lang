// RUN: llvm-mc -filetype=obj -arch=x86 %s | llvm-objdump -d - \
// RUN:                                    | FileCheck %s -check-prefix=WITHRAW
// RUN: llvm-mc -filetype=obj -arch=x86 %s | llvm-objdump -d -no-show-raw-insn - \
// RUN:                                    | FileCheck %s -check-prefix=NORAW

// Expect to find the raw incoding when run with raw output (default), but not
// when run explicitly with -no-show-raw-insn

movl 0, %eax
// WITHRAW: a1 00 00 00 00 movl

// NORAW: movl
// NORAW-NOT: a1 00


