// RUN: llvm-mc -triple x86_64-apple-darwin -x86-asm-syntax=intel %s | FileCheck %s
// rdar://14961158
	.text
	.align 16
	.globl FUNCTION_NAME
	.private_extern	FUNCTION_NAME
FUNCTION_NAME:
	.intel_syntax
	cmp rdi, 1
	jge 1f
// CHECK:	jge	"L11"
	add rdi, 2
1:
// CHECK:	"L11":
	add rdi, 1
	ret
