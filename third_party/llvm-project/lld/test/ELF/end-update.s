// REQUIRES: x86
// Should set the value of the "end" symbol if it is undefined.

// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: ld.lld %t.o -o %t
// RUN: llvm-readobj --sections --symbols %t | FileCheck %s

// CHECK: Sections [
// CHECK:     Name: .bss
// CHECK-NEXT:     Type:
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x202159
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 6
// CHECK: ]
// CHECK: Symbols [
// CHECK:     Name: end
// CHECK-NEXT:     Value: 0x20215F
// CHECK: ]

.global _start,end
.text
_start:
    nop
.bss
    .space 6
