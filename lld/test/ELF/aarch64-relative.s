// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-unknown-freebsd %s -o %t.o
// RUN: ld.lld %t.o -o %t.so -shared
// RUN: llvm-readobj -r %t.so | FileCheck %s
        adrp    x8, .Lfoo

        .data
        .Lfoo:

        .rodata
        .long .Lfoo - .

// CHECK:      Relocations [
// CHECK-NEXT: ]
