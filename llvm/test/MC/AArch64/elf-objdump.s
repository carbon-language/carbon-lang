// 64 bit little endian
// RUN: llvm-mc -filetype=obj -arch=aarch64 -triple aarch64-none-linux-gnu %s -o - | llvm-objdump -d

// We just want to see if llvm-objdump works at all.
// CHECK: .text
