# RUN: llvm-mc %s -triple=x86_64-unknown-linux-gnu -mcpu=corei7 -mattr=+sse2 -asm-instrumentation=address -asan-instrument-assembly | FileCheck %s

	.text
	.globl	mov1b
	.align	16, 0x90
	.type	mov1b,@function
# CHECK-LABEL: mov1b:
#
# CHECK: leaq -128(%rsp), %rsp
# CHECK-NEXT: pushq %rdi
# CHECK-NEXT: leaq (%rsi), %rdi
# CHECK-NEXT: callq __sanitizer_sanitize_load1@PLT
# CHECK-NEXT: popq %rdi
# CHECK-NEXT: leaq 128(%rsp), %rsp
#
# CHECK-NEXT: movb (%rsi), %al
#
# CHECK-NEXT: leaq -128(%rsp), %rsp
# CHECK-NEXT: pushq %rdi
# CHECK-NEXT: leaq (%rdi), %rdi
# CHECK-NEXT: callq __sanitizer_sanitize_store1@PLT
# CHECK-NEXT: popq %rdi
# CHECK-NEXT: leaq 128(%rsp), %rsp
#
# CHECK-NEXT: movb %al, (%rdi)
mov1b:                                  # @mov1b
	.cfi_startproc
# BB#0:
	#APP
	movb	(%rsi), %al
	movb	%al, (%rdi)

	#NO_APP
	retq
.Ltmp0:
	.size	mov1b, .Ltmp0-mov1b
	.cfi_endproc

	.globl	mov16b
	.align	16, 0x90
	.type	mov16b,@function
# CHECK-LABEL: mov16b:
#
# CHECK: leaq -128(%rsp), %rsp
# CHECK-NEXT: pushq %rdi
# CHECK-NEXT: leaq (%rsi), %rdi
# CHECK-NEXT: callq __sanitizer_sanitize_load16@PLT
# CHECK-NEXT: popq %rdi
# CHECK-NEXT: leaq 128(%rsp), %rsp
#
# CHECK-NEXT: movaps (%rsi), %xmm0
#
# CHECK-NEXT: leaq -128(%rsp), %rsp
# CHECK-NEXT: pushq %rdi
# CHECK-NEXT: leaq (%rdi), %rdi
# CHECK-NEXT: callq __sanitizer_sanitize_store16@PLT
# CHECK-NEXT: popq %rdi
# CHECK-NEXT: leaq 128(%rsp), %rsp
#
# CHECK-NEXT: movaps %xmm0, (%rdi)
mov16b:                                 # @mov16b
	.cfi_startproc
# BB#0:
	#APP
	movaps	(%rsi), %xmm0
	movaps	%xmm0, (%rdi)

	#NO_APP
	retq
.Ltmp1:
	.size	mov16b, .Ltmp1-mov16b
	.cfi_endproc


	.ident	"clang version 3.5 "
	.section	".note.GNU-stack","",@progbits
