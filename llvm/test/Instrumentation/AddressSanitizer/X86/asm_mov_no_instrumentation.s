# RUN: llvm-mc %s -triple=x86_64-unknown-linux-gnu -mcpu=corei7 -mattr=+sse2 | FileCheck %s

	.text
	.globl	mov1b
	.align	16, 0x90
	.type	mov1b,@function
# CHECK-LABEL: mov1b
# CHECK-NOT: callq __sanitizer_sanitize_load1@PLT
# CHECK-NOT: callq __sanitizer_sanitize_store1@PLT
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
# CHECK-LABEL: mov16b
# CHECK-NOT: callq __sanitizer_sanitize_load16@PLT
# CHECK-NOT: callq __sanitizer_sanitize_store16@PLT
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
