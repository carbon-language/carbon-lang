// RUN: llvm-mc -triple=thumbv7-none-linux-gnueabi -mcpu=cortex-a8 -filetype=obj < %s | llvm-objdump -arch=thumb -s - | FileCheck %s --check-prefix=CHECK-LE
// RUN: llvm-mc -triple=thumbebv7-none-linux-gnueabi -mcpu=cortex-a8 -filetype=obj < %s | llvm-objdump -arch=thumbeb -s - | FileCheck %s --check-prefix=CHECK-BE
  .syntax unified
  .code 16
  .thumb_func
foo:
  movw r9, :lower16:(_bar)

// CHECK-LE: Contents of section .text:
// CHECK-LE-NEXT: 0000 40f20009
// CHECK-BE: Contents of section .text:
// CHECK-BE-NEXT: 0000 f2400900
