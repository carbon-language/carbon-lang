// RUN: llvm-mc -triple x86_64-unknown-unknown  %s | FileCheck %s

	.intel_syntax
_test:
// CHECK:	movl	$257, -4(%rsp)
	mov	DWORD PTR [RSP - 4], 257

