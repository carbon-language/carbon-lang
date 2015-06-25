// RUN: llvm-mc %s -o %t -filetype=obj -triple=x86_64-pc-linux
// RUN: llvm-nm --print-size %t | FileCheck %s

// CHECK: 0000000000000000 ffffffffffffffff n a
// CHECK: 0000000000000000 0000000000000000 N b

        .section foo
a:
        .size a, 0xffffffffffffffff

        .global b
b:
