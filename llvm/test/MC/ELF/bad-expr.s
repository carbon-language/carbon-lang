// RUN: not llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o /dev/null 2>%t
// RUN: FileCheck --input-file=%t %s

// CHECK: symbol '__executable_start' can not be undefined in a subtraction expression

        .data
x:
        .quad   x-__executable_start
