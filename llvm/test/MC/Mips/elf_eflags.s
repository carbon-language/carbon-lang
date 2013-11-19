; XFAIL: *
// RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux %s -o -| llvm-readobj -h | FileCheck %s


// CHECK: Flags [ (0x50001005)
