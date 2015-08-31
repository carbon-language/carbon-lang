// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/common.s -o %t2
// RUN: lld -flavor gnu2 %t %t2 -o %t3
// RUN: llvm-readobj -t %t3 | FileCheck %s
// REQUIRES: x86

// CHECK:      Name: sym2
// CHECK-NEXT: Value: 0x0
// CHECK-NEXT: Size: 8
// CHECK-NEXT: Binding: Global
// CHECK-NEXT: Type: Object
// CHECK-NEXT: Other: 0
// CHECK-NEXT: Section: Undefined

// CHECK:      Name: sym1
// CHECK-NEXT: Value: 0x0
// CHECK-NEXT: Size: 8
// CHECK-NEXT: Binding: Global
// CHECK-NEXT: Type: Object
// CHECK-NEXT: Other: 0
// CHECK-NEXT: Section: Undefined


.globl _start
_start:

.comm sym1,4,4
.comm sym2,8,4
