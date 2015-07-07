// RUN: llvm-mc %s -o %t -filetype=obj -triple=x86_64-pc-linux
// RUN: llvm-nm --print-size %t | FileCheck %s

// CHECK: 0000000000000000 ffffffffffffffff n a
// CHECK: 0000000000000000 0000000000000000 N b
// CHECK: 0000000000000004 0000000000000004 C c
// CHECK: ffffffffffffffff 0000000000000000 a d

        .section foo
a:
        .size a, 0xffffffffffffffff

        .global b
b:

        .comm c,4,8

d = 0xffffffffffffffff
