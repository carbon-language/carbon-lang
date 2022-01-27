// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i386-pc-linux %s -o %t
// RUN: ld.lld --hash-style=sysv -shared %t -o %t2
// RUN: llvm-readobj --symbols %t2 | FileCheck %s

// The X86 _GLOBAL_OFFSET_TABLE_ is defined at the start of the .got.plt
// section.
.globl  a
.type   a,@object
.comm   a,4,4

.globl  f
.type   f,@function
f:
addl    $_GLOBAL_OFFSET_TABLE_, %eax
movl    a@GOT(%eax), %eax

.global _start
.type _start,@function
_start:
addl    $_GLOBAL_OFFSET_TABLE_, %eax
calll   f@PLT

// CHECK:     Name: _GLOBAL_OFFSET_TABLE_
// CHECK-NEXT:     Value:
// CHECK-NEXT:     Size: 0
// CHECK-NEXT:     Binding: Local (0x0)
// CHECK-NEXT:     Type: None (0x0)
// CHECK-NEXT:     Other [ (0x2)
// CHECK-NEXT:       STV_HIDDEN (0x2)
// CHECK-NEXT:     ]
// CHECK-NEXT:     Section: .got.plt
