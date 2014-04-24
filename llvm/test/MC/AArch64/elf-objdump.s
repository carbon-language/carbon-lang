// 64 bit little endian
// RUN: llvm-mc -filetype=obj -triple aarch64-none-linux-gnu %s -o - | llvm-objdump -d -
// RUN: llvm-mc -filetype=obj -triple arm64-none-linux-gnu %s -o - | llvm-objdump -d -

// We just want to see if llvm-objdump works at all.
// CHECK: .text
