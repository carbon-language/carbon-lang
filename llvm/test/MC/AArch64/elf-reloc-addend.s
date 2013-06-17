// RUN: llvm-mc -triple=aarch64-linux-gnu -filetype=obj -o - %s | llvm-objdump -triple=aarch64-linux-gnu -r - | FileCheck %s

	add x0, x4, #:lo12:sym
// CHECK: 0 R_AARCH64_ADD_ABS_LO12_NC sym
	add x3, x5, #:lo12:sym+1
// CHECK: 4 R_AARCH64_ADD_ABS_LO12_NC sym+1
	add x3, x5, #:lo12:sym-1
// CHECK: 8 R_AARCH64_ADD_ABS_LO12_NC sym-1
