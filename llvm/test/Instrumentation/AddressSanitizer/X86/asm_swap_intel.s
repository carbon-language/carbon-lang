# RUN: llvm-mc %s -x86-asm-syntax=intel -triple=x86_64-unknown-linux-gnu -asm-instrumentation=address | FileCheck %s

	.text
	.globl	swap
	.align	16, 0x90
	.type	swap,@function
# CHECK-LABEL: swap:
#
# CHECK: subq $128, %rsp
# CHECK-NEXT: pushq %rdi
# CHECK-NEXT: leaq (%rcx), %rdi
# CHECK-NEXT: callq __sanitizer_sanitize_load8@PLT
# CHECK-NEXT: popq %rdi
# CHECK-NEXT: addq $128, %rsp
#
# CHECK-NEXT: movq (%rcx), %rax
#
# CHECK-NEXT: subq $128, %rsp
# CHECK-NEXT: pushq %rdi
# CHECK-NEXT: leaq (%rdx), %rdi
# CHECK-NEXT: callq __sanitizer_sanitize_load8@PLT
# CHECK-NEXT: popq %rdi
# CHECK-NEXT: addq $128, %rsp
#
# CHECK-NEXT: movq (%rdx), %rbx
#
# CHECK: subq $128, %rsp
# CHECK-NEXT: pushq %rdi
# CHECK-NEXT: leaq (%rcx), %rdi
# CHECK-NEXT: callq __sanitizer_sanitize_store8@PLT
# CHECK-NEXT: popq %rdi
# CHECK-NEXT: addq $128, %rsp
#
# CHECK-NEXT: movq %rbx, (%rcx)
#
# CHECK-NEXT: subq $128, %rsp
# CHECK-NEXT: pushq %rdi
# CHECK-NEXT: leaq (%rdx), %rdi
# CHECK-NEXT: callq __sanitizer_sanitize_store8@PLT
# CHECK-NEXT: popq %rdi
# CHECK-NEXT: addq $128, %rsp
#
# CHECK-NEXT: movq %rax, (%rdx)
swap:                                   # @swap
	.cfi_startproc
# BB#0:
	push	rbx
.Ltmp0:
	.cfi_def_cfa_offset 16
.Ltmp1:
	.cfi_offset rbx, -16
	mov	rcx, rdi
	mov	rdx, rsi
	#APP


	mov	rax, qword ptr [rcx]
	mov	rbx, qword ptr [rdx]
	mov	qword ptr [rcx], rbx
	mov	qword ptr [rdx], rax

	#NO_APP
	pop	rbx
	ret
.Ltmp2:
	.size	swap, .Ltmp2-swap
	.cfi_endproc


	.ident	"clang version 3.5.0 "
	.section	".note.GNU-stack","",@progbits
