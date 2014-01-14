// RUN: llvm-mc -filetype=obj -triple mipsel-unknown-linux %s -o -| llvm-readobj -t | FileCheck %s


.globl f1
.set micromips
f1:
	nop

.globl f2
.set nomicromips
f2:
	nop

// CHECK-LABEL: Name: f1
// CHECK:       Other: 128
// CHECK-LABEL: Name: f2
// CHECK:       Other: 0
