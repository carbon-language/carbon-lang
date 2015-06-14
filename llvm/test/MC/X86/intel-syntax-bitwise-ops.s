// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=att %s | FileCheck %s

.intel_syntax

// CHECK: andl	$3, %ecx
    and ecx, 1+2
// CHECK: andl	$3, %ecx
    and ecx, 1|2
// CHECK: andl	$3, %ecx
    and ecx, 1*3
// CHECK: andl	$1, %ecx
    and ecx, 1&3
// CHECK: andl	$0, %ecx
    and ecx, (1&2)
// CHECK: andl	$3, %ecx
    and ecx, ((1)|2)
// CHECK: andl	$1, %ecx
    and ecx, 1&2+3
// CHECK: addl	$4938, %eax
    add eax, 9876 >> 1
// CHECK: addl	$19752, %eax
    add eax, 9876 << 1
// CHECK: addl	$5, %eax
    add eax, 6 ^ 3
