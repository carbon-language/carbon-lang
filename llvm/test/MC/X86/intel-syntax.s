// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=intel %s | FileCheck %s

_test:
	xor	EAX, EAX
	ret

_main:
// CHECK:	movl	$257, -4(%rsp)
	mov	DWORD PTR [RSP - 4], 257
// CHECK:	movl	$258, 4(%rsp)
	mov	DWORD PTR [RSP + 4], 258
// CHECK:	movq	$123, -16(%rsp)
	mov	QWORD PTR [RSP - 16], 123
// CHECK:	movb	$97, -17(%rsp)
	mov	BYTE PTR [RSP - 17], 97
// CHECK:	movl	-4(%rsp), %eax
	mov	EAX, DWORD PTR [RSP - 4]
// CHECK:	movq    (%rsp), %rax
	mov     RAX, QWORD PTR [RSP]
// CHECK:	movl	$-4, -4(%rsp)
	mov	DWORD PTR [RSP - 4], -4
// CHECK:	movq	0, %rcx
	mov	RCX, QWORD PTR [0]
// CHECK:	movl	-24(%rsp,%rax,4), %eax	
	mov	EAX, DWORD PTR [RSP + 4*RAX - 24]
// CHECK:	movb	%dil, (%rdx,%rcx)
	mov	BYTE PTR [RDX + RCX], DIL
// CHECK:	movzwl	2(%rcx), %edi
	movzx	EDI, WORD PTR [RCX + 2]
// CHECK:	callq	_test
	call	_test
	ret
