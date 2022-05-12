// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
// RUN: ld.lld -shared %t -o %t2
// RUN: llvm-readobj -S --symbols %t2 | FileCheck %s

// The X86_64 _GLOBAL_OFFSET_TABLE_ is defined at the start of the .got.plt
// section.
.globl  a
.type   a,@object
.comm   a,4,4

.globl  f
.type   f,@function
f:
movq	a@GOTPCREL(%rip), %rax

.global _start
.type _start,@function
_start:
callq	f@PLT
.data
.long _GLOBAL_OFFSET_TABLE_ - .

// CHECK:     Name: .got.plt
// CHECK:     Address: [[ADDR:.*]]
// CHECK:     Name: _GLOBAL_OFFSET_TABLE_
// CHECK-NEXT:     Value: [[ADDR]]
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local
// CHECK-NEXT:     Type: None (0x0)
// CHECK-NEXT:     Other [
// CHECK-NEXT:       STV_HIDDEN
// CHECK-NEXT:     ]
// CHECK-NEXT:     Section: .got.plt
