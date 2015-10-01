// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: lld -flavor gnu2 %t.o -shared -soname=bar -o %t.so
// RUN: lld -flavor gnu2 %t.o %t.so -o %t
// RUN: llvm-readobj --dynamic-table %t | FileCheck %s

// CHECK:  0x0000000000000001 NEEDED               SharedLibrary (bar)

.global _start
_start:
