// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/resolution-shared.s -o %t2.o
// RUN: ld.lld %t2.o -o %t2.so -shared
// RUN: ld.lld %t.o %t2.so -o %t3 -shared
// RUN: llvm-readobj --symbols %t3 | FileCheck %s

        .weak foo
foo:

// CHECK:      Symbol {
// CHECK:        Name: foo
// CHECK-NEXT:   Value:
// CHECK-NEXT:   Size:
// CHECK-NEXT:   Binding: Weak
