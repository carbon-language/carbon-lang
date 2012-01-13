// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel %s | FileCheck %s

// CHECK:	movl	$257, -4(%rsp)
	mov	DWORD PTR [RSP - 4], 257
// CHECK:	movq	$123, -16(%rsp)
	mov	QWORD PTR [RSP - 16], 123
// CHECK:	movb	$97, -17(%rsp)
	mov	BYTE PTR [RSP - 17], 97
// CHECK:	movl	-4(%rsp), %eax
	mov	EAX, DWORD PTR [RSP - 4]
