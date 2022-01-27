// RUN: llvm-mc -triple x86_64-unknown-unknown -x86-asm-syntax=att %s | FileCheck %s

	.intel_syntax
_test:
// CHECK:	movl	$257, -4(%rsp)
	mov	DWORD PTR [RSP - 4], 257
    .att_syntax
// CHECK:	movl	$257, -4(%rsp)
    movl $257, -4(%rsp)

_test2:
.intel_syntax noprefix
	mov	DWORD PTR [RSP - 4], 255
// CHECK:	movl	$255, -4(%rsp)
.att_syntax prefix
	movl $255, -4(%rsp)
// CHECK:	movl	$255, -4(%rsp)

_test3:
fadd 
// CHECK: faddp %st, %st(1)
fmul
// CHECK: fmulp %st, %st(1)
fsub
// CHECK: fsubp %st, %st(1)
fsubr
// CHECK: fsubrp %st, %st(1)
fdiv
// CHECK: fdivp %st, %st(1)
fdivr
// CHECK: fdivrp %st, %st(1)
