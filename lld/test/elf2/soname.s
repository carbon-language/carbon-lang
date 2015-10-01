// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: lld -flavor gnu2 %t.o %p/Inputs/soname.so -o %t
// RUN: llvm-readobj --dynamic-table %t | FileCheck %s

// CHECK:  0x0000000000000001 NEEDED               SharedLibrary (bar)

.global _start
_start:
