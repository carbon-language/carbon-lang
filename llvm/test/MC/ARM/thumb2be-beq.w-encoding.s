// RUN: llvm-mc -triple=thumbv7-none-linux-gnueabi -mcpu=cortex-a8 -filetype=obj < %s | llvm-objdump -arch=thumb -s - | FileCheck %s --check-prefix=CHECK-LE
// RUN: llvm-mc -triple=thumbebv7-none-linux-gnueabi -mcpu=cortex-a8 -filetype=obj < %s | llvm-objdump -arch=thumbeb -s - | FileCheck %s --check-prefix=CHECK-BE
  .syntax unified
  .code 16
  .thumb_func
foo:
  beq.w   bar

// CHECK-LE: Contents of section .text:
// CHECK-LE-NEXT: 0000 3ff4feaf
// CHECK-BE: Contents of section .text:
// CHECK-BE-NEXT: 0000 f43faffe
