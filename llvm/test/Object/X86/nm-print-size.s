// RUN: llvm-mc %s -o %t -filetype=obj -triple=x86_64-pc-linux
// RUN: llvm-nm --print-size %t | FileCheck %s

// CHECK: 0000000000000000 ffffffffffffffff t a

a:
        .size a, 0xffffffffffffffff
