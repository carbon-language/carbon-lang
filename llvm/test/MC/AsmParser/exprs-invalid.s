// RUN: not llvm-mc -triple i386-unknown-unknown %s 2> %t
// RUN: FileCheck -input-file %t %s

        .text
a:
        .data
// CHECK: expected relocatable expression
        .long -(0 + a)
