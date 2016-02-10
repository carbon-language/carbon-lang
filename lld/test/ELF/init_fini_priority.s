// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
// RUN: ld.lld %t -o %t.exe
// RUN: llvm-objdump -s %t.exe | FileCheck %s
// REQUIRES: x86

.globl _start
_start:
  nop

.section .init_array, "aw", @init_array
  .align 8
  .byte 1
.section .init_array.100, "aw", @init_array
  .long 2
.section .init_array.5, "aw", @init_array
  .byte 3

.section .fini_array, "aw", @fini_array
  .align 8
  .byte 4
.section .fini_array.100, "aw", @fini_array
  .long 5
.section .fini_array.5, "aw", @fini_array
  .byte 6

// CHECK:      Contents of section .init_array:
// CHECK-NEXT: 03020000 00000000 01
// CHECK:      Contents of section .fini_array:
// CHECK-NEXT: 06050000 00000000 04
