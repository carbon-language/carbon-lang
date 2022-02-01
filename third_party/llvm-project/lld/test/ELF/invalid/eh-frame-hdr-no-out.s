// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
// RUN: not ld.lld --eh-frame-hdr %t -o /dev/null 2>&1 | FileCheck %s

// CHECK: error: corrupted .eh_frame: FDE version 1 or 3 expected, but got 2

.section .eh_frame,"a",@unwind
  .byte 0x08
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x00
  .byte 0x02
  .byte 0x00
  .byte 0x00
  .byte 0x00
